"""
Milvus 向量索引构建模块

负责将分块后的文档转换为向量，并构建 Milvus 集合以供相似度检索。

核心流程：
    Document 列表 → (embed_documents) → 向量矩阵
                                  ↓
                          (create_collection + insert) → Milvus 集合
                                  ↓
                           (create_index: HNSW-COSINE) → 近似最近邻索引

Milvus 负责存储和检索，HuggingFaceEmbeddings (BGE-small-zh-v1.5) 负责向量化，
两者通过 collection 串联。本模块提供集合的完整生命周期管理（创建/加载/删除）
和相似度搜索接口。

外部依赖：
    - Milvus server（HTTP 地址由 host:port 指定）
    - HuggingFaceEmbeddings (BAAI/bge-small-zh-v1.5)：中文文本向量化
"""

import logging
import time
from typing import List, Dict, Any, Optional

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import numpy as np

logger = logging.getLogger(__name__)


class MilvusIndexConstructionModule:
    """Milvus 向量索引构建模块

    职责：
        - 建立 Milvus 客户端连接并管理集合生命周期（创建/加载/删除）
        - 使用 HuggingFaceEmbeddings (BGE-small-zh-v1.5) 将文本转换为向量
        - 构建 HNSW-COSINE 近似最近邻索引以支持快速相似度搜索

    设计要点：
        - 集合字段覆盖 chunk_id / parent_id / category / cuisine_type 等，
          支持按菜系、分类等条件过滤检索。
        - 使用 HNSW (M=16, efConstruction=200) + COSINE 相似度，
          在中文场景下兼顾检索精度与查询速度。

    Public API:
        - build_vector_index(chunks)     → 将全量文档构建为向量集合（含建索引）
        - similarity_search(query, k)    → 对查询文本执行向量相似度检索
        - add_documents(new_chunks)      → 向已有集合追加新文档
    """

    def __init__(self,
                 host: str = "localhost",
                 port: int = 19530,
                 collection_name: str = "cooking_knowledge",
                 dimension: int = 512,
                 model_name: str = "BAAI/bge-small-zh-v1.5"):
        """初始化 Milvus 索引构建模块。

        Args:
            host: Milvus 服务器地址（默认 localhost）
            port: Milvus 服务器端口（默认 19530）
            collection_name: Milvus 集合名称（默认 "cooking_knowledge"）
            dimension: 向量维度（BGE-small-zh-v1.5 输出 512 维）
            model_name: HuggingFace 嵌入模型名称（默认 BGE-small-zh-v1.5）
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.model_name = model_name

        self.client = None       # Milvus 客户端实例
        self.embeddings = None   # HuggingFace 嵌入模型实例
        self.collection_created = False  # 标记集合是否已创建

        self._setup_client()      # 初始化 Milvus 客户端
        self._setup_embeddings()  # 初始化嵌入模型

    def _safe_truncate(self, text: str, max_length: int) -> str:
        """安全截取字符串，处理 None 值。

        Args:
            text: 输入文本（可能为 None）
            max_length: 最大截取长度

        Returns:
            截断后的字符串（None → ""）
        """
        if text is None:
            return ""
        return str(text)[:max_length]

    def _setup_client(self):
        """初始化 Milvus 客户端连接。

        使用 MilvusClient(uri="http://host:port") 建立 HTTP 连接，
        并列出当前所有集合以验证连通性。
        """
        try:
            self.client = MilvusClient(
                uri=f"http://{self.host}:{self.port}"
            )
            logger.info(f"已连接到Milvus服务器: {self.host}:{self.port}")

            # 列出当前所有集合，验证连接是否成功。
            collections = self.client.list_collections()
            logger.info(f"连接成功，当前集合: {collections}")

        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise

    def _setup_embeddings(self):
        """初始化 HuggingFace 嵌入模型（BGE-small-zh-v1.5）。

        model_kwargs={'device': 'cpu'} 表示使用 CPU 推理，
        encode_kwargs={'normalize_embeddings': True} 表示输出归一化向量（便于 COSINE 计算）。
        """
        logger.info(f"正在初始化嵌入模型: {self.model_name}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},           # 使用 CPU 推理（部署时可改为 GPU）
            encode_kwargs={'normalize_embeddings': True}  # 归一化 → COSINE 距离可直接比较
        )

        logger.info("嵌入模型初始化完成")

    def _create_collection_schema(self) -> CollectionSchema:
        """创建 Milvus 集合的数据模式（schema）。

        定义集合中每个字段的类型和约束：
            - id:     VARCHAR(150)，主键，存储 chunk 的唯一标识
            - vector: FLOAT_VECTOR(dim=512)，嵌入向量（余弦相似度）
            - text:   VARCHAR(15000)，chunk 的原始文本内容（供调试展示）
            - node_id / recipe_name: 菜谱相关元信息字段
            - category / cuisine_type / difficulty: 筛选过滤字段

        Returns:
            CollectionSchema 对象（用于 create_collection）
        """
        # 定义集合字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=150, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=15000),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="recipe_name", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="cuisine_type", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="difficulty", dtype=DataType.INT64),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100)
        ]

        # 创建集合模式（描述：中式烹饪知识图谱向量集合）
        schema = CollectionSchema(
            fields=fields,
            description="中式烹饪知识图谱向量集合"
        )

        return schema

    def create_collection(self, force_recreate: bool = False) -> bool:
        """创建 Milvus 集合。

        Args:
            force_recreate: 是否强制删除已有集合并重新创建（默认 False）

        Returns:
            True → 成功（已存在或新建完成）；False → 异常中断
        """
        try:
            # 检查集合是否已存在
            if self.client.has_collection(self.collection_name):
                if force_recreate:
                    logger.info(f"删除已存在的集合: {self.collection_name}")
                    self.client.drop_collection(self.collection_name)
                else:
                    logger.info(f"集合 {self.collection_name} 已存在")
                    self.collection_created = True
                    return True

            # 创建集合：使用预定义的 schema，余弦相似度度量，强一致性级别。
            schema = self._create_collection_schema()

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                metric_type="COSINE",  # 使用余弦相似度（归一化向量等价于内积）
                consistency_level="Strong"  # 强一致性：写入后立即可见（适合 RAG）
            )

            logger.info(f"成功创建集合: {self.collection_name}")
            self.collection_created = True

            return True

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False

    def create_index(self) -> bool:
        """创建向量索引（HNSW-COSINE）。

        使用 HNSW (Hierarchical Navigable Small World) 算法构建近似最近邻索引，
        参数 M=16（每节点邻居数）、efConstruction=200（构建时的搜索宽度），
        在索引质量和构建时间之间取得平衡。

        Returns:
            True → 成功；False → 异常中断（集合未创建时抛出 ValueError）
        """
        try:
            if not self.collection_created:
                raise ValueError("请先创建集合")

            # 使用 prepare_index_params 构建正确的 IndexParams 对象
            index_params = self.client.prepare_index_params()

            # 添加向量字段索引（HNSW-COSINE）
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",                   # 层次导航小世界算法（近似最近邻）
                metric_type="COSINE",                 # 余弦相似度度量
                params={
                    "M": 16,                          # 每节点最大邻居数（越大越精确，构建越慢）
                    "efConstruction": 200             # 构建时的搜索宽度（越大索引质量越高）
                }
            )

            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )

            logger.info("向量索引创建成功")
            return True

        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return False

    def build_vector_index(self, chunks: List[Document]) -> bool:
        """构建完整的 Milvus 向量索引（创建集合 → 生成向量 → 插入数据 → 建索引）。

        这是批量构建流程，将一组 Document 分块转换为向量并写入 Milvus：

        Args:
            chunks: 文档分块列表（由 GraphDataPreparationModule.chunk_documents 生成）

        Returns:
            True → 构建成功；False → 任意步骤失败
        """
        logger.info(f"正在构建Milvus向量索引，文档数量: {len(chunks)}...")

        if not chunks:
            raise ValueError("文档块列表不能为空")

        try:
            # 1. 创建集合（force_recreate=True：每次构建都重建，保证数据一致性）
            if not self.create_collection(force_recreate=True):
                return False

            # 2. 使用嵌入模型将所有 chunk 的文本转换为向量矩阵
            logger.info("正在生成向量embeddings...")
            texts = [chunk.page_content for chunk in chunks]
            vectors = self.embeddings.embed_documents(texts)

            # 3. 组装插入数据：每个 chunk 对应一个实体（id + vector + text + metadata）
            entities = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                entity = {
                    "id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{i}"), 150),
                    "vector": vector,                     # 归一化后的 512 维向量
                    "text": self._safe_truncate(chunk.page_content, 15000),
                    "node_id": self._safe_truncate(chunk.metadata.get("node_id", ""), 100),
                    "recipe_name": self._safe_truncate(chunk.metadata.get("recipe_name", ""), 300),
                    "node_type": self._safe_truncate(chunk.metadata.get("node_type", ""), 100),
                    "category": self._safe_truncate(chunk.metadata.get("category", ""), 100),
                    "cuisine_type": self._safe_truncate(chunk.metadata.get("cuisine_type", ""), 200),
                    "difficulty": int(chunk.metadata.get("difficulty", 0)),
                    "doc_type": self._safe_truncate(chunk.metadata.get("doc_type", ""), 50),
                    "chunk_id": self._safe_truncate(chunk.metadata.get("chunk_id", f"chunk_{i}"), 150),
                    "parent_id": self._safe_truncate(chunk.metadata.get("parent_id", ""), 100)
                }
                entities.append(entity)

            # 4. 批量插入数据（每批 100 条）
            logger.info("正在插入向量数据...")
            batch_size = 100
            for i in range(0, len(entities), batch_size):
                batch = entities[i:i + batch_size]
                self.client.insert(
                    collection_name=self.collection_name,
                    data=batch
                )
                logger.info(f"已插入 {min(i + batch_size, len(entities))}/{len(entities)} 条数据")

            # 5. 创建向量索引（HNSW）
            if not self.create_index():
                return False

            # 6. 加载集合到内存（使索引可用）
            self.client.load_collection(self.collection_name)
            logger.info("集合已加载到内存")

            # 7. 等待索引构建完成（HNSW 异步构建，需短暂等待）
            logger.info("等待索引构建完成...")
            time.sleep(2)

            logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
            return True

        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
            return False

    def add_documents(self, new_chunks: List[Document]) -> bool:
        """向已有集合追加新文档（增量更新）。

        适用于需要动态扩充知识库的场景：不重建整个集合，仅插入新 chunk。

        Args:
            new_chunks: 新的文档分块列表

        Returns:
            True → 成功；False → 异常中断
        """
        if not self.collection_created:
            raise ValueError("请先构建向量索引")

        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")

        try:
            # 生成向量（使用同一个嵌入模型）
            texts = [chunk.page_content for chunk in new_chunks]
            vectors = self.embeddings.embed_documents(texts)

            # 组装插入数据（与 build_vector_index 中的格式一致）
            entities = []
            for i, (chunk, vector) in enumerate(zip(new_chunks, vectors)):
                entity = {
                    "id": self._safe_truncate(chunk.metadata.get("chunk_id", f"new_chunk_{i}_{int(time.time())}"), 150),
                    "vector": vector,
                    "text": self._safe_truncate(chunk.page_content, 15000),
                    "node_id": self._safe_truncate(chunk.metadata.get("node_id", ""), 100),
                    "recipe_name": self._safe_truncate(chunk.metadata.get("recipe_name", ""), 300),
                    "node_type": self._safe_truncate(chunk.metadata.get("node_type", ""), 100),
                    "category": self._safe_truncate(chunk.metadata.get("category", ""), 100),
                    "cuisine_type": self._safe_truncate(chunk.metadata.get("cuisine_type", ""), 200),
                    "difficulty": int(chunk.metadata.get("difficulty", 0)),
                    "doc_type": self._safe_truncate(chunk.metadata.get("doc_type", ""), 50),
                    "chunk_id": self._safe_truncate(chunk.metadata.get("chunk_id", f"new_chunk_{i}_{int(time.time())}"), 150),
                    "parent_id": self._safe_truncate(chunk.metadata.get("parent_id", ""), 100)
                }
                entities.append(entity)

            # 插入数据（单批写入）
            self.client.insert(
                collection_name=self.collection_name,
                data=entities
            )

            logger.info("新文档添加完成")
            return True

        except Exception as e:
            logger.error(f"添加新文档失败: {e}")
            return False

    def similarity_search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """相似度搜索：将查询文本转换为向量，在 Milvus 中查找 Top-K 相似结果。

        Args:
            query: 查询文本（如"如何做红烧肉？"）
            k: 返回结果数量（默认 5）
            filters: 可选的过滤条件字典，支持 str/int/list 类型的字段过滤

        Returns:
            搜索结果列表，每个元素包含：
                - id:       匹配文档的 chunk_id
                - score:    COSINE 相似度分数（值越大越相似）
                - text:     chunk 的原始文本内容
                - metadata: 菜谱名称、分类、难度等元信息

        Note:
            COSINE 距离中，值越大表示查询向量与文档向量的方向越接近（即语义越相似）。
        """
        if not self.collection_created:
            raise ValueError("请先构建或加载向量索引")

        try:
            # 将查询文本转换为 512 维向量（使用同一个嵌入模型）
            query_vector = self.embeddings.embed_query(query)

            # 构建过滤表达式（如有 filter 条件）
            filter_expr = ""
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if isinstance(value, str):
                        filter_conditions.append(f'{key} == "{value}"')
                    elif isinstance(value, (int, float)):
                        filter_conditions.append(f'{key} == {value}')
                    elif isinstance(value, list):
                        # 支持 IN 操作（如 category IN ["川菜", "鲁菜"]）
                        if all(isinstance(v, str) for v in value):
                            value_str = '", "'.join(value)
                            filter_conditions.append(f'{key} in ["{value_str}"]')
                        else:
                            value_str = ', '.join(map(str, value))
                            filter_conditions.append(f'{key} in [{value_str}]')

                if filter_conditions:
                    filter_expr = " and ".join(filter_conditions)

            # 构建搜索参数（HNSW ef=64：查询时的搜索宽度，越大越精确）
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}
            }

            # 构建搜索 kwargs（避免重复传递参数）
            search_kwargs = {
                "collection_name": self.collection_name,
                "data": [query_vector],           # 单个查询向量（Milvus 格式）
                "anns_field": "vector",           # 在 vector 字段上执行 ANN 搜索
                "limit": k,                       # 返回 Top-K 结果
                "output_fields": ["text", "node_id", "recipe_name", "node_type",
                                "category", "cuisine_type", "difficulty", "doc_type",
                                "chunk_id", "parent_id"],  # 只返回需要的字段（减少网络开销）
                "search_params": search_params
            }

            # 仅在存在过滤条件时添加 filter 参数
            if filter_expr:
                search_kwargs["filter"] = filter_expr

            # 执行搜索（返回的结果格式: [hit_list]，hit_list[0] 对应第一个查询向量）
            results = self.client.search(**search_kwargs)

            # 处理结果：将 Milvus 原始格式转换为统一结构
            formatted_results = []
            if results and len(results) > 0:
                for hit in results[0]:  # results[0] 对应我们发送的第一个（也是唯一的）查询向量
                    result = {
                        "id": hit["id"],
                        "score": hit["distance"],  # COSINE 距离：值越大相似度越高（即方向越接近）
                        "text": hit["entity"]["text"],
                        "metadata": {
                            "node_id": hit["entity"]["node_id"],
                            "recipe_name": hit["entity"]["recipe_name"],
                            "node_type": hit["entity"]["node_type"],
                            "category": hit["entity"]["category"],
                            "cuisine_type": hit["entity"]["cuisine_type"],
                            "difficulty": hit["entity"]["difficulty"],
                            "doc_type": hit["entity"]["doc_type"],
                            "chunk_id": hit["entity"]["chunk_id"],
                            "parent_id": hit["entity"]["parent_id"]
                        }
                    }
                    formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息（记录数、索引构建进度等）。

        Returns:
            包含 row_count / index_building_progress 等的字典。
        """
        try:
            if not self.collection_created:
                return {"error": "集合未创建"}

            stats = self.client.get_collection_stats(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "row_count": stats.get("row_count", 0),
                "index_building_progress": stats.get("index_building_progress", 0),
                "stats": stats
            }

        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            return {"error": str(e)}

    def delete_collection(self) -> bool:
        """删除集合（用于重建知识库时清理旧数据）。

        Returns:
            True → 成功；False → 异常中断
        """
        try:
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
                logger.info(f"集合 {self.collection_name} 已删除")
                self.collection_created = False
                return True
            else:
                logger.info(f"集合 {self.collection_name} 不存在")
                return True

        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False

    def has_collection(self) -> bool:
        """检查集合是否存在。

        Returns:
            True → 存在；False → 不存在或异常中断
        """
        try:
            return self.client.has_collection(self.collection_name)
        except Exception as e:
            logger.error(f"检查集合存在性失败: {e}")
            return False

    def load_collection(self) -> bool:
        """将集合加载到内存（使索引可用，方可执行搜索）。

        Returns:
            True → 成功；False → 异常中断（集合不存在时）
        """
        try:
            if not self.client.has_collection(self.collection_name):
                logger.error(f"集合 {self.collection_name} 不存在")
                return False

            self.client.load_collection(self.collection_name)
            self.collection_created = True
            logger.info(f"集合 {self.collection_name} 已加载到内存")
            return True

        except Exception as e:
            logger.error(f"加载集合失败: {e}")
            return False

    def close(self):
        """关闭连接（Milvus 客户端内部会自动释放资源，此方法仅用于日志记录）。"""
        if hasattr(self, 'client') and self.client:
            # Milvus 客户端不需要显式关闭，但记录日志方便排查。
            logger.info("Milvus连接已关闭")

    def __del__(self):
        """析构函数：确保资源释放。"""
        self.close()