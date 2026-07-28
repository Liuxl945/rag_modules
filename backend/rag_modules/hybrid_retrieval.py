"""
混合检索模块 — 基于 LightRAG 的双层检索范式（实体级 + 主题级）

结合 BM25 (jieba 分词)、向量检索与图键值索引，使用 RRF (Reciprocal Rank Fusion) 融合三路结果。

核心特点：
    1. 双层检索范式（实体级 + 主题级，基于图键值索引）
    2. BM25 关键词检索（jieba 分词 + 停用词过滤）
    3. Milvus 向量检索（含图结构一跳邻居扩展）
    4. RRF (Reciprocal Rank Fusion) 融合三路结果

检索流程：
    query → [LLM关键词提取] → 实体级关键词 + 主题级关键词
                              ↓                          ↓
                    (entity_level)              (topic_level)
                        ↓                           ↓
         ┌────────── 图键值索引（实体 + 关系）  ──────────┐
         │                                              │
         ↓                                              ↓
      BM25                                          Milvus 向量检索
         │                                              ↑
         └────────────── RRF 三路融合 ──────────────────┘
                              ↓
                        Top-K 文档

外部依赖：
    - jieba: 中文分词器（BM25 使用）
    - rank_bm25: BM25Okapi（标准词频-逆文档频率加权）
    - Milvus / GraphDataPreparationModule: 向量检索与图数据源
"""

import json
import logging
import os
import pickle
import hashlib
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from neo4j import GraphDatabase
from .graph_indexing import GraphIndexingModule, EntityKeyValue, RelationKeyValue

logger = logging.getLogger(__name__)

# 中文停用词表：助词 / 连词 / 疑问词 / 人称代词 / 语气词 / 动词修饰语
# 不引入第三方停用词包，按烹饪问答场景手动挑选（覆盖测试集高频虚词）。
_CHINESE_STOPWORDS = set("""
的 了 和 是 在 我 有 就 不 也 都 还 这 那 一 个 与 及 等 上 下 中 为 以 于 从 把 被 让 使 又 而 但 或
什么 怎么 如何 哪些 哪个 哪里 谁 多少 几 你 他 她 它 我们 他们 她们 它们
请问 请 想 要 需要 能 可以 应该 会 啊 呢 吧 嘛 吗 哦 呀 哈
之 其 此 该 即 各 每 些 种 类 时 后 前 里 外 内 间 已经 正在 一些 一下
""".split())

# RRF 融合的常数 k：Cormack et al. 2009 默认值（用于平滑排名贡献）
_RRF_K = 60


@dataclass
class RetrievalResult:
    """检索结果数据结构：封装单一检索通道的返回结果。

    Attributes:
        content: 该条目的文本内容（由检索逻辑组装）
        node_id: 来源节点在图数据库中的唯一标识
        node_type: 节点类型（Recipe / Ingredient）
        relevance_score: 相关性得分（精确匹配 → 0.9，分类匹配 → 0.85）
        retrieval_level: 检索层级（entity / topic），用于下游区分处理
        metadata: 扩展元信息（实体名、匹配关键词等）
    """
    content: str
    node_id: str
    node_type: str
    relevance_score: float
    retrieval_level: str  # 'entity' 或 'topic'
    metadata: Dict[str, Any]


class HybridRetrievalModule:
    """混合检索模块

    核心特点：
        1. 双层检索范式（实体级 + 主题级）：基于图键值索引，分别命中具体实体和抽象概念
        2. BM25 关键词检索：jieba 分词 + 停用词过滤，精确匹配用户查询中的实体
        3. 向量检索（Milvus）：语义相似度搜索，捕获同义/近义表达
        4. RRF (Reciprocal Rank Fusion)：三路结果的加权融合，综合利用三种检索的优势

    Public API:
        - hybrid_search(query, top_k) → 三路召回 → RRF 融合 → Top-K 文档
        - dual_level_retrieval(query, top_k) → 图键值双层检索 → Document 列表
        - vector_search_enhanced(query, top_k) → Milvus 向量检索 + 邻居扩展
        - bm25_search(query, top_k) → BM25 关键词检索
    """

    def __init__(self, config, milvus_module, data_module, llm_client):
        self.config = config
        self.milvus_module = milvus_module  # Milvus 向量索引模块
        self.data_module = data_module      # Neo4j 数据准备模块
        self.llm_client = llm_client        # LLM 客户端（用于关键词提取等智能操作）
        self.driver = None                  # Neo4j 数据库连接

        # BM25 索引 + 原始文档（按索引位置对齐）
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus_docs: List[Document] = []

        # 图索引模块（封装 LightRAG K,V 结构）
        self.graph_indexing = GraphIndexingModule(config, llm_client)
        self.graph_indexed = False

        # 最近一次 hybrid_search 的通道统计（dict | None），供 Web API 透传到前端
        # 展示三路召回贡献：candidates=各路候选数，final=融合后入选数。
        self.last_hybrid_stats = None

    # 各检索通道原始分数在 Document.metadata 中的字段名（用于 RRF 时保留每源原始分）
    _CHANNEL_SCORE_KEY = {
        "dual_level": "relevance_score",  # 双层检索规则分（0.75-0.95）
        "vector": "score",                # Milvus 余弦相似度（0-1）
        "bm25": "bm25_score",             # BM25 TF-IDF 分（无界，>0）
    }

    def initialize(self, chunks: List[Document]):
        """初始化检索系统：BM25 建索引 + 图索引构建。

        Args:
            chunks: 文档分块列表（由 GraphDataPreparationModule.chunk_documents 生成）

        初始化步骤：
            1. 连接 Neo4j（用于后续图查询）
            2. 构建 BM25 索引（基于 jieba 分词 + 停用词过滤）
            3. 构建图键值索引（实体 KV + 关系 KV）
            4. 建立父文档映射（供 RRF 融合后的「父文档回填」使用）
        """
        logger.info("初始化混合检索模块...")

        # 1. 连接 Neo4j（用于图查询和邻居扩展）
        self.driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password)
        )

        # 2. 初始化 BM25（基于 jieba 分词 + 中文停用词过滤）
        #    分词结果会持久化到磁盘，启动时优先读缓存，避免每次重启都重复 jieba 分词。
        if chunks:
            self.bm25_corpus_docs = list(chunks)
            # 计算语料指纹：文档数量/内容/顺序任一变化，指纹即变，缓存自动失效。
            fingerprint = self._compute_corpus_fingerprint(chunks)
            tokenized_corpus = self._load_tokenized_cache(fingerprint)
            if tokenized_corpus is None:
                logger.info("未命中 BM25 分词缓存，开始 jieba 分词...")
                tokenized_corpus = [self._tokenize_chinese(d.page_content) for d in chunks]
                self._save_tokenized_cache(fingerprint, tokenized_corpus)
            # 用分词结果构建 BM25 索引（算 idf，比 jieba 分词快得多）
            self.bm25 = BM25Okapi(tokenized_corpus)
            avg_tokens = sum(len(t) for t in tokenized_corpus) / max(1, len(tokenized_corpus))
            logger.info(
                f"BM25(jieba+stopwords) 索引构建完成，文档数: {len(chunks)}，"
                f"平均 token 数: {avg_tokens:.1f}"
            )

        # 3. 构建图索引（为双层检索提供 K,V 结构）
        self._build_graph_index()

        # 4. 建立父文档映射（由分块前的 data_module.documents 懒建一次），
        #    供后续 RRF 融合后做「父文档回填」使用（仅前 top_n 条命中时替换）。
        self._parent_doc_map = self._build_parent_doc_map()
        logger.info(f"父文档映射构建完成，菜谱文档数: {len(self._parent_doc_map)}")

    @staticmethod
    def _tokenize_chinese(text: str) -> List[str]:
        """使用 jieba 精确分词，并过滤停用词 / 空白 / 单字符。

        Args:
            text: 输入文本

        Returns:
            过滤后的分词列表（去除停用词、空白和单字符，降低噪声）
        """
        if not text:
            return []
        tokens = jieba.lcut(text)
        return [
            t for t in tokens
            if t.strip() and t not in _CHINESE_STOPWORDS and not t.isspace()
        ]

    # ========== BM25 分词结果持久化 ==========
    # jieba 分词是 CPU 密集操作，文档量大时启动耗时主要花在这里。
    # 将 tokenized_corpus pickle 到磁盘，启动时优先读缓存，
    # 命中则跳过分词，只做 BM25Okapi 构建（算 idf，比分词快得多）。

    def _get_bm25_cache_dir(self) -> str:
        """获取 BM25 分词缓存目录。

        优先使用 config.bm25_cache_dir；未配置时回退到模块同级目录下的 .bm25_cache/。
        目录不存在时自动创建。
        """
        cache_dir = getattr(self.config, "bm25_cache_dir", "") or ""
        if not cache_dir:
            cache_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), ".bm25_cache"
            )
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    @staticmethod
    def _compute_corpus_fingerprint(chunks: List[Document]) -> str:
        """计算语料指纹：基于每个 chunk 的 chunk_id + page_content 的 md5。

        文档数量 / 内容 / 顺序任一变化，指纹都会变，用于判断分词缓存是否失效。
        """
        h = hashlib.md5()
        for d in chunks:
            cid = str(d.metadata.get("chunk_id", ""))
            content_hash = hashlib.md5(d.page_content.encode("utf-8")).hexdigest()
            h.update(f"{cid}:{content_hash}\n".encode("utf-8"))
        return h.hexdigest()

    def _load_tokenized_cache(self, fingerprint: str) -> Optional[List[List[str]]]:
        """加载缓存的分词结果（指纹不匹配或读取失败均视为未命中）。

        Args:
            fingerprint: 当前语料的指纹

        Returns:
            tokenized_corpus（命中时）；None（未命中或失败时）
        """
        cache_path = os.path.join(
            self._get_bm25_cache_dir(),
            f"{self.config.milvus_collection_name}_{fingerprint[:16]}.pkl",
        )
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            if data.get("fingerprint") != fingerprint:
                logger.info("BM25 分词缓存指纹不匹配，丢弃重建")
                return None
            logger.info(f"命中 BM25 分词缓存: {cache_path}")
            return data.get("tokenized_corpus")
        except Exception as e:
            logger.warning(f"加载 BM25 分词缓存失败，将重新分词: {e}")
            return None

    def _save_tokenized_cache(self, fingerprint: str, tokenized_corpus: List[List[str]]) -> None:
        """保存分词结果到磁盘缓存（失败不影响主流程）。"""
        cache_path = os.path.join(
            self._get_bm25_cache_dir(),
            f"{self.config.milvus_collection_name}_{fingerprint[:16]}.pkl",
        )
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "fingerprint": fingerprint,
                        "tokenized_corpus": tokenized_corpus,
                        "chunk_count": len(tokenized_corpus),
                        "method": "jieba+stopwords",
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            logger.info(f"BM25 分词结果已缓存: {cache_path}")
        except Exception as e:
            logger.warning(f"保存 BM25 分词缓存失败（不影响运行）: {e}")

    def _build_graph_index(self):
        """构建图键值索引。

        从 data_module 中读取已加载的 Recipe / Ingredient / CookingStep 节点，
        调用 GraphIndexingModule 创建实体和关系的 K,V 结构。

        Note:
            此步骤仅在首次调用时执行，后续通过 self.graph_indexed 标记跳过。
        """
        if self.graph_indexed:
            return

        logger.info("开始构建图索引...")

        try:
            # 获取图中所有实体（Recipe / Ingredient / CookingStep）
            recipes = self.data_module.recipes
            ingredients = self.data_module.ingredients
            cooking_steps = self.data_module.cooking_steps

            # 创建实体键值对（名称 → 描述）
            self.graph_indexing.create_entity_key_values(recipes, ingredients, cooking_steps)

            # 从 Neo4j 提取关系三元组（source_id, relation_type, target_id）
            relationships = self._extract_relationships_from_graph()
            self.graph_indexing.create_relation_key_values(relationships)

            # 去重优化（合并重复实体和关系）
            self.graph_indexing.deduplicate_entities_and_relations()

            self.graph_indexed = True
            stats = self.graph_indexing.get_statistics()
            logger.info(f"图索引构建完成: {stats}")

        except Exception as e:
            logger.error(f"构建图索引失败: {e}")

    def _extract_relationships_from_graph(self) -> List[Tuple[str, str, str]]:
        """从 Neo4j 图中提取关系三元组 (source_id, relation_type, target_id)。

        使用 Cypher 查询 `MATCH (source)-[r]->(target)` 获取所有边，
        限制最多 1000 条关系（避免内存溢出）。

        Returns:
            (source_id, relation_type, target_id) 三元组列表
        """
        relationships = []

        try:
            with self.driver.session() as session:
                query = """
                MATCH (source)-[r]->(target)
                WHERE source.nodeId >= '200000000' OR target.nodeId >= '200000000'
                RETURN source.nodeId as source_id, type(r) as relation_type, target.nodeId as target_id
                LIMIT 1000
                """
                result = session.run(query)

                for record in result:
                    relationships.append((
                        record["source_id"],
                        record["relation_type"],
                        record["target_id"]
                    ))

        except Exception as e:
            logger.error(f"提取图关系失败: {e}")

        return relationships

    def extract_query_keywords(self, query: str) -> Tuple[List[str], List[str]]:
        """使用 LLM 提取查询关键词，分为实体级和主题级两层。

        通过调用 LLM 的智能理解能力，将用户的自然语言查询分解为：
            - entity_keywords: 具体的食材、菜品名、工具等（如"鸡胸肉""平底锅"）
            - topic_keywords: 抽象概念、烹饪主题、饮食风格（如"减肥""川菜"）

        Args:
            query: 用户的自然语言查询（如"推荐几个减肥菜"）

        Returns:
            (entity_keywords, topic_keywords) 元组
        """
        prompt = f"""
        作为烹饪知识助手，请分析以下查询并提取关键词，分为两个层次：

        查询：{query}

        提取规则：
        1. 实体级关键词：具体的食材、菜品名称、工具、品牌等有形实体
           - 例如：鸡胸肉、西兰花、红烧肉、平底锅、老干妈
           - 对于抽象查询，推测相关的具体食材/菜品

        2. 主题级关键词：抽象概念、烹饪主题、饮食风格、营养特点等
           - 例如：减肥、低热量、川菜、素食、下饭菜、快手菜
           - 排除动作词：推荐、介绍、制作、怎么做等

        示例：
        查询："推荐几个减肥菜"
        {{
            "entity_keywords": ["鸡胸肉", "西兰花", "水煮蛋", "胡萝卜", "黄瓜"],
            "topic_keywords": ["减肥", "低热量", "高蛋白", "低脂"]
        }}

        查询："川菜有什么特色"
        {{
            "entity_keywords": ["麻婆豆腐", "宫保鸡丁", "水煮鱼", "辣椒", "花椒"],
            "topic_keywords": ["川菜", "麻辣", "香辣", "下饭菜"]
        }}

        请严格按照JSON格式返回，不要包含多余的文字：
        {{
            "entity_keywords": ["实体1", "实体2", ...],
            "topic_keywords": ["主题1", "主题2", ...]
        }}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # 低温度确保输出稳定、格式正确
                max_tokens=500
            )

            result = json.loads(response.choices[0].message.content.strip())
            entity_keywords = result.get("entity_keywords", [])
            topic_keywords = result.get("topic_keywords", [])

            logger.info(f"关键词提取完成 - 实体级: {entity_keywords}, 主题级: {topic_keywords}")
            return entity_keywords, topic_keywords

        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            # 降级方案：简单的空格分割（不依赖 LLM）
            keywords = query.split()
            return keywords[:3], keywords[3:6] if len(keywords) > 3 else keywords

    def entity_level_retrieval(self, entity_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """实体级检索：专注于具体实体和关系。

        使用图索引的键值对结构进行精确匹配：
            1. 遍历每个实体关键词，在 entity_kv_store 中查找匹配的实体
            2. 对匹配到的实体，扩展其一跳邻居（关联食材/步骤等）
            3. 组装增强内容，赋予较高相关性得分 (0.9)

        Args:
            entity_keywords: 实体级关键词列表（如 ["鸡胸肉", "西兰花"]）
            top_k: 返回结果数量上限

        Returns:
            RetrievalResult 列表（按相关性得分降序排列）
        """
        results = []

        # 1. 使用图索引进行实体检索（精确匹配）
        for keyword in entity_keywords:
            # 在键值索引中查找匹配的实体
            entities = self.graph_indexing.get_entities_by_key(keyword)

            for entity in entities:
                # 获取该实体的一跳邻居信息（关联的食材/步骤等）
                neighbors = self._get_node_neighbors(entity.metadata["node_id"], max_neighbors=2)

                # 构建增强内容：原始实体信息 + 邻居信息的拼接输出
                enhanced_content = entity.value_content
                if neighbors:
                    enhanced_content += f"\n相关信息: {', '.join(neighbors)}"

                results.append(RetrievalResult(
                    content=enhanced_content,
                    node_id=entity.metadata["node_id"],
                    node_type=entity.entity_type,
                    relevance_score=0.9,  # 精确匹配得分较高（实体级优先）
                    retrieval_level="entity",
                    metadata={
                        "entity_name": entity.entity_name,
                        "entity_type": entity.entity_type,
                        "index_keys": entity.index_keys,
                        "matched_keyword": keyword  # 记录匹配到的原始关键词
                    }
                ))

        # 2. 如果图索引结果不足，使用 Neo4j 全文索引进行补充检索
        if len(results) < top_k:
            neo4j_results = self._neo4j_entity_level_search(entity_keywords, top_k - len(results))
            results.extend(neo4j_results)

        # 3. 按相关性排序并返回 Top-K
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.info(f"实体级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]

    def _neo4j_entity_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        """Neo4j 全文索引补充检索（实体级降级方案）。

        当图键值索引未命中足够结果时，使用 Neo4j 的全文索引 (fulltext index)
        进行模糊匹配。注意得分为原始分数 * 0.7（补充检索优先级低于精确匹配）。

        Args:
            keywords: 关键词列表
            limit: 最多返回结果数

        Returns:
            RetrievalResult 列表（补充检索得分较低）
        """
        results = []

        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                CALL db.index.fulltext.queryNodes('recipe_fulltext_index', keyword + '*')
                YIELD node, score
                WHERE node:Recipe
                RETURN
                    node.nodeId as node_id,
                    node.name as name,
                    node.description as description,
                    labels(node) as labels,
                    score
                ORDER BY score DESC
                LIMIT $limit
                """

                result = session.run(cypher_query, {
                    "keywords": keywords,
                    "limit": limit
                })

                for record in result:
                    content_parts = []
                    if record["name"]:
                        content_parts.append(f"菜品: {record['name']}")
                    if record["description"]:
                        content_parts.append(f"描述: {record['description']}")

                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=record["node_id"],
                        node_type="Recipe",
                        relevance_score=float(record["score"]) * 0.7,  # 补充检索得分较低
                        retrieval_level="entity",
                        metadata={
                            "name": record["name"],
                            "labels": record["labels"],
                            "source": "neo4j_fallback"  # 标记来源为补充检索
                        }
                    ))

        except Exception as e:
            logger.error(f"Neo4j补充检索失败: {e}")

        return results

    def topic_level_retrieval(self, topic_keywords: List[str], top_k: int = 5) -> List[RetrievalResult]:
        """主题级检索：专注于广泛主题和概念。

        使用图索引的关系键值对结构进行主题检索，以及基于实体分类的补充检索：
            1. 在 relation_kv_store 中查找包含主题关键词的关系（如「食材搭配」对应 REQUIRES）
            2. 在 entity_kv_store 中查找分类匹配的 Recipe（如 category="川菜"）
            3. 若结果不足，降级使用 Neo4j Cypher 按 category / cuisineType / tags 字段匹配

        Args:
            topic_keywords: 主题级关键词列表（如 ["减肥", "川菜"]）
            top_k: 返回结果数量上限

        Returns:
            RetrievalResult 列表（按相关性得分降序排列）
        """
        results = []

        # 1. 使用图索引进行关系/主题检索（通过关系键值对的 index_keys 匹配）
        for keyword in topic_keywords:
            # 在关系键值索引中查找匹配的关系（如「食材搭配」→ REQUIRES）
            relations = self.graph_indexing.get_relations_by_key(keyword)

            for relation in relations:
                # 获取该关系关联的源实体和目标实体信息
                source_entity = self.graph_indexing.entity_kv_store.get(relation.source_entity)
                target_entity = self.graph_indexing.entity_kv_store.get(relation.target_entity)

                if source_entity and target_entity:
                    # 构建丰富的主题内容：包含关系描述、源/目标实体信息。
                    content_parts = [
                        f"主题: {keyword}",
                        relation.value_content,
                        f"相关菜品: {source_entity.entity_name}",
                        f"相关信息: {target_entity.entity_name}"
                    ]

                    # 如果源实体是 Recipe，额外添加菜品详情（提升输出质量）
                    if source_entity.entity_type == "Recipe":
                        newline = '\n'
                        content_parts.append(f"菜品详情: {source_entity.value_content.split(newline)[0]}")

                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=relation.source_entity,  # 以源实体（通常是 Recipe）为 ID
                        node_type=source_entity.entity_type,
                        relevance_score=0.95,  # 主题匹配得分（关系级优先）
                        retrieval_level="topic",
                        metadata={
                            "relation_id": relation.relation_id,
                            "relation_type": relation.relation_type,
                            "source_name": source_entity.entity_name,
                            "target_name": target_entity.entity_name,
                            "matched_keyword": keyword,
                            "index_keys": relation.index_keys
                        }
                    ))

        # 2. 使用实体的分类信息进行主题检索（如 category="川菜" → 相关 Recipe）
        for keyword in topic_keywords:
            entities = self.graph_indexing.get_entities_by_key(keyword)
            for entity in entities:
                if entity.entity_type == "Recipe":
                    # 构建分类主题内容：菜名 + 菜品详情。
                    content_parts = [
                        f"主题分类: {keyword}",
                        entity.value_content
                    ]

                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=entity.metadata["node_id"],
                        node_type=entity.entity_type,
                        relevance_score=0.85,  # 分类匹配得分（低于关系级）
                        retrieval_level="topic",
                        metadata={
                            "entity_name": entity.entity_name,
                            "entity_type": entity.entity_type,
                            "matched_keyword": keyword,
                            "source": "category_match"  # 标记来源为分类匹配
                        }
                    ))

        # 3. 如果结果不足，使用 Neo4j Cypher 按属性字段补充检索
        if len(results) < top_k:
            neo4j_results = self._neo4j_topic_level_search(topic_keywords, top_k - len(results))
            results.extend(neo4j_results)

        # 4. 按相关性排序并返回 Top-K
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.info(f"主题级检索完成，返回 {len(results)} 个结果")
        return results[:top_k]

    def _neo4j_topic_level_search(self, keywords: List[str], limit: int) -> List[RetrievalResult]:
        """Neo4j 主题级检索补充（按 category / cuisineType / tags 匹配）。

        当图键值索引的主题检索未命中足够结果时，使用 Cypher 查询
        `WHERE r.category CONTAINS keyword OR r.cuisineType CONTAINS keyword` 进行模糊匹配。

        Args:
            keywords: 主题关键词列表
            limit: 最多返回结果数

        Returns:
            RetrievalResult 列表（得分较低，来源标记为 neo4j_fallback）
        """
        results = []

        try:
            with self.driver.session() as session:
                cypher_query = """
                UNWIND $keywords as keyword
                MATCH (r:Recipe)
                WHERE r.category CONTAINS keyword
                   OR r.cuisineType CONTAINS keyword
                   OR r.tags CONTAINS keyword
                WITH r, keyword
                OPTIONAL MATCH (r)-[:REQUIRES]->(i:Ingredient)
                WITH r, keyword, collect(i.name)[0..3] as ingredients
                RETURN
                    r.nodeId as node_id,
                    r.name as name,
                    r.category as category,
                    r.cuisineType as cuisine_type,
                    r.difficulty as difficulty,
                    ingredients,
                    keyword as matched_keyword
                ORDER BY r.difficulty ASC, r.name
                LIMIT $limit
                """

                result = session.run(cypher_query, {
                    "keywords": keywords,
                    "limit": limit
                })

                for record in result:
                    content_parts = []
                    content_parts.append(f"菜品: {record['name']}")

                    if record["category"]:
                        content_parts.append(f"分类: {record['category']}")
                    if record["cuisine_type"]:
                        content_parts.append(f"菜系: {record['cuisine_type']}")
                    if record["difficulty"]:
                        content_parts.append(f"难度: {record['difficulty']}")

                    if record["ingredients"]:
                        ingredients_str = ', '.join(record["ingredients"][:3])
                        content_parts.append(f"主要食材: {ingredients_str}")

                    results.append(RetrievalResult(
                        content='\n'.join(content_parts),
                        node_id=record["node_id"],
                        node_type="Recipe",
                        relevance_score=0.75,  # 补充检索得分（最低优先级）
                        retrieval_level="topic",
                        metadata={
                            "name": record["name"],
                            "category": record["category"],
                            "cuisine_type": record["cuisine_type"],
                            "difficulty": record["difficulty"],
                            "matched_keyword": record["matched_keyword"],
                            "source": "neo4j_fallback"  # 标记来源为补充检索
                        }
                    ))

        except Exception as e:
            logger.error(f"Neo4j主题级检索失败: {e}")

        return results

    def dual_level_retrieval(self, query: str, top_k: int = 5) -> List[Document]:
        """双层检索：结合实体级和主题级检索，输出 Document 列表。

        这是「图键值双层检索」的主接口：
            1. LLM 提取实体级 + 主题级关键词（extract_query_keywords）
            2. 分别执行实体级检索和主题级检索
            3. 合并、去重（按 node_id）、按相关性排序

        Args:
            query: 用户的自然语言查询
            top_k: 返回结果数量上限

        Returns:
            Document 列表（metadata 包含 search_type="dual_level"）
        """
        logger.info(f"开始双层检索: {query}")

        # 1. 使用 LLM 提取查询关键词（实体级 + 主题级）
        entity_keywords, topic_keywords = self.extract_query_keywords(query)

        # 2. 分别执行双层检索
        entity_results = self.entity_level_retrieval(entity_keywords, top_k)
        topic_results = self.topic_level_retrieval(topic_keywords, top_k)

        # 3. 结果合并和排序（实体级通常更精确，优先保留高相关度条目）
        all_results = entity_results + topic_results

        # 4. 去重：按 node_id 去重，保留相关性得分最高的条目。
        seen_nodes = set()
        unique_results = []

        for result in sorted(all_results, key=lambda x: x.relevance_score, reverse=True):
            if result.node_id not in seen_nodes:
                seen_nodes.add(result.node_id)
                unique_results.append(result)

        # 5. 转换为 Document 格式（供下游 RRF 融合使用）
        documents = []
        for result in unique_results[:top_k]:
            # 确保 recipe_name 字段正确设置（从 metadata 中提取或默认）
            recipe_name = result.metadata.get("name") or result.metadata.get("entity_name", "未知菜品")

            doc = Document(
                page_content=result.content,
                metadata={
                    "node_id": result.node_id,
                    "node_type": result.node_type,
                    "retrieval_level": result.retrieval_level,  # entity / topic
                    "relevance_score": result.relevance_score,
                    "recipe_name": recipe_name,  # 确保有 recipe_name 字段（供展示和去重使用）
                    "search_type": "dual_level",  # 设置搜索类型（供 RRF 融合标记来源）
                    **result.metadata  # 展开原始 metadata（matched_keyword, source 等）
                }
            )
            documents.append(doc)

        logger.info(f"双层检索完成，返回 {len(documents)} 个文档")
        return documents

    def vector_search_enhanced(self, query: str, top_k: int = 5) -> List[Document]:
        """增强的向量检索：结合图信息（一跳邻居扩展）。

        在 Milvus 原始搜索结果基础上，对每个命中的 Recipe 节点
        查询其关联的食材/步骤等邻居信息，并拼接到原文中。

        Args:
            query: 查询文本
            top_k: 返回结果数量上限

        Returns:
            Document 列表（metadata 包含 score 和 search_type="vector_enhanced"）
        """
        try:
            # 使用 Milvus 进行向量检索（返回 Top-K*2 候选，留出扩展空间）
            vector_docs = self.milvus_module.similarity_search(query, k=top_k*2)

            # 用图信息增强结果并转换为 Document 对象
            enhanced_docs = []
            for result in vector_docs:
                # 从 Milvus 结果创建 Document 对象
                content = result.get("text", "")
                metadata = result.get("metadata", {})
                node_id = metadata.get("node_id")

                if node_id:
                    # 从图中获取该节点的一跳邻居信息（食材/步骤等）
                    neighbors = self._get_node_neighbors(node_id)
                    if neighbors:
                        # 将邻居信息拼接到原文末尾（丰富上下文）
                        neighbor_info = f"\n相关信息: {', '.join(neighbors[:3])}"
                        content += neighbor_info

                # 确保 recipe_name 字段正确设置（供去重和展示使用）
                recipe_name = metadata.get("recipe_name", "未知菜品")

                # 调试：打印向量得分（便于分析检索效果）
                vector_score = result.get("score", 0.0)
                logger.debug(f"向量检索得分: {recipe_name} = {vector_score}")

                # 创建 Document 对象（包含 score 和 search_type）
                doc = Document(
                    page_content=content,
                    metadata={
                        **metadata,
                        "recipe_name": recipe_name,  # 确保有 recipe_name 字段
                        "score": vector_score,       # 原始 COSINE 相似度分数
                        "search_type": "vector_enhanced"  # 标记来源为增强向量检索
                    }
                )
                enhanced_docs.append(doc)

            return enhanced_docs[:top_k]

        except Exception as e:
            logger.error(f"增强向量检索失败: {e}")
            return []

    def _get_node_neighbors(self, node_id: str, max_neighbors: int = 3) -> List[str]:
        """获取指定节点的邻居名称列表（用于扩展检索结果）。

        使用 Cypher 查询 `MATCH (n {nodeId: $node_id})-[r]-(neighbor)`
        获取该节点的所有关联实体名称，限制返回数量。

        Args:
            node_id: 目标节点的 nodeId
            max_neighbors: 最大返回数量

        Returns:
            邻居节点名称列表（失败时返回空列表）
        """
        try:
            with self.driver.session() as session:
                query = """
                MATCH (n {nodeId: $node_id})-[r]-(neighbor)
                RETURN neighbor.name as name
                LIMIT $limit
                """
                result = session.run(query, {"node_id": node_id, "limit": max_neighbors})
                return [record["name"] for record in result if record["name"]]
        except Exception as e:
            logger.error(f"获取邻居节点失败: {e}")
            return []

    def bm25_search(self, query: str, top_k: int = 5) -> List[Document]:
        """BM25 检索：基于 jieba 分词的关键词精确匹配。

        使用预构建的 BM25Okapi 索引，对查询文本分词后计算 TF-IDF 加权分数，
        按分数降序返回 Top-K 文档。

        Args:
            query: 查询文本
            top_k: 返回结果数量上限

        Returns:
            Document 列表（metadata 包含 bm25_score，供调试和分数级融合使用）

        Note:
            BM25 分数 ≤ 0 的文档视为无关（IDF/TF 均无贡献），不进入结果。
        """
        if self.bm25 is None or not self.bm25_corpus_docs:
            logger.warning("BM25 索引未初始化，bm25_search 返回空")
            return []

        # 对查询文本进行分词（使用相同的停用词过滤逻辑）
        tokenized_query = self._tokenize_chinese(query)
        if not tokenized_query:
            logger.debug(f"BM25 query 分词为空，跳过: {query}")
            return []

        # 计算所有文档的 BM25 分数，按降序取 Top-K
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        docs: List[Document] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                # BM25 分数 ≤ 0 视为无关（IDF/TF 全无贡献），不进结果。
                continue
            src = self.bm25_corpus_docs[idx]
            recipe_name = (
                src.metadata.get("recipe_name")
                or src.metadata.get("name")
                or "未知菜品"
            )
            doc = Document(
                page_content=src.page_content,
                metadata={
                    **src.metadata,
                    "recipe_name": recipe_name,
                    "search_method": "bm25",    # 标记来源为 BM25
                    "search_type": "bm25",      # 统一 search_type 标记
                    "bm25_score": score,        # 原始 BM25 分数（供调试与未来分数级融合使用）
                }
            )
            docs.append(doc)

        logger.info(f"BM25 检索完成，返回 {len(docs)} 个文档（query tokens={tokenized_query}）")
        return docs

    @staticmethod
    def _rrf_merge(
        ranked_lists: List[Tuple[str, List[Document]]],
        top_k: int,
        k: int = _RRF_K,
    ) -> List[Document]:
        """Reciprocal Rank Fusion (RRF)：多路召回结果的加权融合。

        核心公式：score(d) = Σ_i 1 / (k + best_rank_i(d))

        其中：
            - score(d) 是文档 d 的 RRF 融合分数（越高越相关）
            - best_rank_i(d) 是文档 d 在第 i 路召回中的最佳排名（rank 越小越好）
            - k = 60 是平滑常数（Cormack et al. 2009），避免排名差异过大的问题

        Args:
            ranked_lists: 多路召回结果，每个元素为 (source_name, ranked_docs)
                         — docs 按相关度降序排列，source_name 用于标记来源（"dual_level"/"vector"/"bm25"）
            top_k: 最终返回的文档数量
            k: RRF 平滑常数，默认 60（Cormack et al. 2009）

        去重与 canonical doc 策略：
            - 去重 key：优先使用 node_id，若缺失则回退到 page_content[:200] 的 MD5 hash
            - 同 source 内同 doc_id 多次命中（如一道菜的多个 chunk 共享 recipe.nodeId）：
                · 算分只取该 source 内最佳 rank（最小 rank），避免重复加分
                · 命中 chunk 数另存到 rrf_chunk_hits，供后续分析
            - canonical doc（最终展示给 LLM 的 page_content）：
                · 选全局最小 rank 对应的 chunk；rank 相同时按 ranked_lists 顺序优先

        返回的 Document 是新对象，不会 mutate 输入 list 里的 Document。
        """
        # doc_id → source_name → 该 source 内最佳 rank（用于 RRF 算分）
        best_rank_per_source: Dict[str, Dict[str, int]] = {}
        # doc_id → source_name → 该 source 内命中的 chunk 次数（信息存档）
        chunk_hits_per_source: Dict[str, Dict[str, int]] = {}
        # doc_id → (global_best_rank, source_priority, doc) — 选 canonical doc
        # doc_id → source_name → 该 source 命中时的原始分数（供前端分数对比）
        raw_scores_per_source: Dict[str, Dict[str, float]] = {}
        best_doc_info: Dict[str, Tuple[int, int, Document]] = {}

        for source_priority, (source_name, ranked_docs) in enumerate(ranked_lists):
            for rank, doc in enumerate(ranked_docs, start=1):
                node_id = doc.metadata.get("node_id")
                # 去重 key：优先用 node_id，缺失时用 page_content[:200] 的 hash
                doc_id = (
                    str(node_id) if node_id is not None
                    else f"hash::{hashlib.md5(doc.page_content[:200].encode('utf-8')).hexdigest()}"
                )

                if doc_id not in best_rank_per_source:
                    best_rank_per_source[doc_id] = {}
                    chunk_hits_per_source[doc_id] = {}
                    raw_scores_per_source[doc_id] = {}

                curr_best = best_rank_per_source[doc_id].get(source_name)
                # 如果是第一次出现，或者当前 rank 比记录的更小（更靠前），则更新。
                is_best = curr_best is None or rank < curr_best
                if is_best:
                    best_rank_per_source[doc_id][source_name] = rank
                    # 同时记录该通道命中时的原始分数（与最佳 rank 同一份 doc 对应）
                    score_key = HybridRetrievalModule._CHANNEL_SCORE_KEY.get(source_name)
                    if score_key:
                        raw = doc.metadata.get(score_key)
                        if raw is not None:
                            try:
                                raw_scores_per_source[doc_id][source_name] = float(raw)
                            except (TypeError, ValueError):
                                pass

                # 记录该 source 内命中 chunk 的次数（用于后续分析）
                chunk_hits_per_source[doc_id][source_name] = (
                    chunk_hits_per_source[doc_id].get(source_name, 0) + 1
                )

                # 选 canonical doc：取全局最小 rank；rank 相同时按 ranked_lists 顺序优先。
                new_key = (rank, source_priority)
                if (
                    doc_id not in best_doc_info
                    or new_key < (best_doc_info[doc_id][0], best_doc_info[doc_id][1])
                ):
                    best_doc_info[doc_id] = (rank, source_priority, doc)

        # 每个 source 只用 best rank 算一次 RRF 贡献（避免重复加分）
        rrf_scores: Dict[str, float] = {
            doc_id: sum(1.0 / (k + r) for r in source_ranks.values())
            for doc_id, source_ranks in best_rank_per_source.items()
        }

        # 按 RRF 分数降序排列（得分越高越相关）
        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True
        )

        # 组装最终结果：选 canonical doc，添加 RRF 元信息
        merged: List[Document] = []
        for doc_id in sorted_ids[:top_k]:
            _, _, source_doc = best_doc_info[doc_id]
            # 浅 copy metadata，避免 mutate 上游 Document（RRF 不应修改原始对象）
            new_metadata = dict(source_doc.metadata)
            new_metadata["rrf_score"] = rrf_scores[doc_id]       # 融合后分数
            new_metadata["rrf_sources"] = list(best_rank_per_source[doc_id].keys())  # 被哪些路召回
            new_metadata["rrf_ranks"] = dict(best_rank_per_source[doc_id])  # 每路最佳 rank
            new_metadata["rrf_raw_scores"] = dict(raw_scores_per_source[doc_id])  # 每路原始分（分数对比用）
            new_metadata["rrf_chunk_hits"] = dict(chunk_hits_per_source[doc_id])  # 每路命中 chunk 次数
            new_metadata["final_score"] = rrf_scores[doc_id]      # 最终综合分数
            merged.append(Document(
                page_content=source_doc.page_content,
                metadata=new_metadata,
            ))

        return merged

    def _build_parent_doc_map(self) -> Dict[str, Document]:
        """构建父文档映射：{str(node_id): 整篇菜谱 Document}。

        由分块前的 data_module.documents（完整菜谱文档）懒建一次，
        供 _attach_parent_documents 在 RRF 融合后做「父文档回填」使用。

        Returns:
            {node_id: 完整菜谱 Document} 字典
        """
        docs = getattr(self.data_module, "documents", None) or []
        m: Dict[str, Document] = {}
        for d in docs:
            nid = d.metadata.get("node_id")
            if nid is not None:
                m[str(nid)] = d
        return m

    def rebuild_bm25(self):
        """从当前 data_module.chunks 重建 BM25 索引（上传新文档后调用）。

        rank_bm25.BM25Okapi 不支持增量添加，上传菜谱后需从全量 chunks 重建。
        利用已有的分词缓存机制（指纹匹配自动跳过重复分词）。
        """
        chunks = self.data_module.chunks or []
        if not chunks:
            logger.warning("rebuild_bm25: 无 chunks，跳过")
            return
        self.bm25_corpus_docs = list(chunks)
        fingerprint = self._compute_corpus_fingerprint(self.bm25_corpus_docs)
        tokenized_corpus = self._load_tokenized_cache(fingerprint)
        if tokenized_corpus is None:
            logger.info("BM25 重建：未命中分词缓存，开始 jieba 分词...")
            tokenized_corpus = [self._tokenize_chinese(d.page_content) for d in self.bm25_corpus_docs]
            self._save_tokenized_cache(fingerprint, tokenized_corpus)
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 索引重建完成，文档数: {len(self.bm25_corpus_docs)}")

    def add_recipe_to_graph_index(
        self,
        recipe_node,
        parsed,
        recipe_id: str,
        ingredient_ids: Dict[str, str],
    ):
        """增量添加单个上传菜谱到图 KV 索引。

        Args:
            recipe_node: GraphNode（刚加入 data_module.recipes 的新菜谱节点）
            parsed: ParsedRecipe 解析结果
            recipe_id: 菜谱 nodeId
            ingredient_ids: 食材 name -> nodeId 映射（含复用的和新建的）
        """
        gi = self.graph_indexing

        # 1. 添加 Recipe 实体
        recipe_value = (
            f"菜品名称: {recipe_node.name}\n"
            f"分类: 用户上传\n"
            f"难度: {parsed.difficulty}星\n"
            f"食材数: {len(parsed.ingredients)}\n"
            f"步骤数: {len(parsed.steps)}"
        )
        recipe_entity = EntityKeyValue(
            entity_name=recipe_node.name,
            index_keys=[recipe_node.name, f"{recipe_node.name}的做法"],
            value_content=recipe_value,
            entity_type="Recipe",
            metadata={"node_id": recipe_id, "properties": recipe_node.properties},
        )
        gi.entity_kv_store[recipe_id] = recipe_entity
        for key in recipe_entity.index_keys:
            if recipe_id not in gi.key_to_entities[key]:
                gi.key_to_entities[key].append(recipe_id)

        # 2. 添加食材实体和 REQUIRES 关系
        for ing in parsed.ingredients:
            ing_nid = ingredient_ids.get(ing.name)
            if not ing_nid:
                continue
            # 仅当该食材尚未在 KV 存储中时新增（复用的跳过）
            if ing_nid not in gi.entity_kv_store:
                ing_entity = EntityKeyValue(
                    entity_name=ing.name,
                    index_keys=[ing.name],
                    value_content=f"食材: {ing.name}\n分类: 未知",
                    entity_type="Ingredient",
                    metadata={"node_id": ing_nid},
                )
                gi.entity_kv_store[ing_nid] = ing_entity
                gi.key_to_entities[ing.name].append(ing_nid)

            # REQUIRES 关系
            rel_id = f"rel_{recipe_id}_REQUIRES_{ing_nid}"
            ing_amount = None
            for ia in parsed.ingredient_amounts:
                if ia.name == ing.name:
                    ing_amount = ia
                    break
            rel_content = f"{recipe_node.name} 需要食材 {ing.name}"
            if ing_amount and ing_amount.amount is not None and ing_amount.unit:
                rel_content += f"，用量: {ing_amount.amount}{ing_amount.unit}"
            rel = RelationKeyValue(
                relation_id=rel_id,
                index_keys=["REQUIRES", "食材搭配", "烹饪原料", ing.name],
                value_content=rel_content,
                relation_type="REQUIRES",
                source_entity=recipe_id,
                target_entity=ing_nid,
                metadata={"source_name": recipe_node.name, "target_name": ing.name},
            )
            gi.relation_kv_store[rel_id] = rel
            for key in rel.index_keys:
                gi.key_to_relations[key].append(rel_id)

        # 3. 添加步骤实体和 CONTAINS_STEP 关系
        for i, step_text in enumerate(parsed.steps):
            step_id = f"{recipe_id}_step_{i}"
            step_name = f"步骤{i+1}"
            step_entity = EntityKeyValue(
                entity_name=step_name,
                index_keys=[step_name, f"{recipe_node.name}{step_name}"],
                value_content=f"{step_name}（{recipe_node.name}）: {step_text}",
                entity_type="CookingStep",
                metadata={"node_id": step_id, "description": step_text, "stepNumber": i + 1},
            )
            gi.entity_kv_store[step_id] = step_entity
            for key in step_entity.index_keys:
                gi.key_to_entities[key].append(step_id)

            step_rel_id = f"rel_{recipe_id}_CONTAINS_STEP_{step_id}"
            step_rel = RelationKeyValue(
                relation_id=step_rel_id,
                index_keys=["CONTAINS_STEP", "烹饪步骤", f"{recipe_node.name}步骤"],
                value_content=f"{recipe_node.name} {step_name}: {step_text}",
                relation_type="CONTAINS_STEP",
                source_entity=recipe_id,
                target_entity=step_id,
                metadata={"source_name": recipe_node.name, "step_order": i},
            )
            gi.relation_kv_store[step_rel_id] = step_rel
            for key in step_rel.index_keys:
                gi.key_to_relations[key].append(step_rel_id)

        # 4. BELONGS_TO_CATEGORY 关系（到"用户上传"分类）
        cat_rel_id = f"rel_{recipe_id}_BELONGS_TO_CATEGORY_upload"
        cat_rel = RelationKeyValue(
            relation_id=cat_rel_id,
            index_keys=["BELONGS_TO_CATEGORY", "菜品分类", "用户上传"],
            value_content=f"{recipe_node.name} 属于分类 用户上传",
            relation_type="BELONGS_TO_CATEGORY",
            source_entity=recipe_id,
            target_entity="cat:用户上传",
            metadata={"source_name": recipe_node.name, "category": "用户上传"},
        )
        gi.relation_kv_store[cat_rel_id] = cat_rel
        for key in cat_rel.index_keys:
            gi.key_to_relations[key].append(cat_rel_id)

        logger.info(
            f"图 KV 索引增量更新完成: +1 Recipe, +{len(parsed.ingredients)} Ingredients(含复用), "
            f"+{len(parsed.steps)} Steps"
        )

    def _attach_parent_documents(self, docs: List[Document]) -> List[Document]:
        """RRF 融合后，对前 top_n 条命中进行「父文档回填」。

        说明：
            RRF 融合后返回的文档是「分块后的 chunk」，内容可能不完整
            （如只包含某个步骤）。此时用完整菜谱文档替换前 top_n 条的 page_content，
            使 LLM 生成答案时有更完整的上下文。

        逻辑：
            - 仅对前 parent_doc_top_n 条且能在映射中找到父菜谱的进行替换
            - 其余保持原样不变（不改顺序/数量/排名）
            - 被替换的造新 Document，未替换的直接传原对象

        Args:
            docs: RRF 融合后的文档列表（按 RRF 分数排序）

        Returns:
            处理后的文档列表
        """
        if getattr(self, "_parent_doc_map", None) is None:
            self._parent_doc_map = self._build_parent_doc_map()
        pmap = self._parent_doc_map
        if not pmap:
            logger.warning("父文档映射为空（data_module.documents 未就绪），父文档回填未生效，仍然使用原chunk填充上下文")
            return docs

        # 读取配置中的父文档回填参数（默认前3条、最多4000字符）
        top_n = getattr(self.config, "parent_doc_top_n", 3)
        max_chars = getattr(self.config, "parent_doc_max_chars", 4000)

        out: List[Document] = []
        for i, doc in enumerate(docs):
            # 超过 top_n 的条目不替换，直接传递原对象（节省内存）。
            if i >= top_n:
                out.append(doc)
                continue

            nid = doc.metadata.get("node_id")
            key = str(nid if nid is not None else doc.metadata.get("parent_id"))
            parent = pmap.get(key)

            if parent is None:
                # 找不到对应父文档（可能是新菜谱、未构建的节点），保持原样。
                out.append(doc)
                continue

            pc = parent.page_content or ""
            if len(pc) > max_chars:
                # 超过上限时截断并添加提示信息
                pc = pc[:max_chars] + "…（父文档已截断）"

            # 用完整菜谱文档替换 chunk 的 page_content（保留原始 metadata）
            out.append(Document(page_content=pc, metadata=dict(doc.metadata)))

        return out

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Document]:
        """混合检索：三路召回（图键值双层 + 向量 + BM25）→ RRF 融合。

        这是「传统混合检索」的主接口：

            query
              ↓
          ┌──────────┬───────────┬──────────┐
          │ 图键值双层  │ Milvus向量  │ BM25关键词 │
          └──────────┴───────────┴──────────┘
                      ↓ RRF 融合（k=60） ↓
                  Top-K 文档 → 父文档回填

        Args:
            query: 用户的自然语言查询
            top_k: 最终返回的文档数量

        Returns:
            Document 列表（metadata 包含 rrf_score, final_score, search_method 等）
        """
        logger.info(f"开始混合检索（dual + vector + bm25, RRF k={_RRF_K}）: {query}")

        # 每路给 RRF 留够候选空间（否则三路各自前 top_k 容易没交集，融合退化）。
        # candidate_k = max(top_k * 2, 10) 确保至少有 10 个候选进入融合。
        candidate_k = max(top_k * 2, 10)

        # 三路召回
        dual_docs = self.dual_level_retrieval(query, candidate_k)    # 图键值双层检索（实体级 + 主题级）
        vector_docs = self.vector_search_enhanced(query, candidate_k) # Milvus 向量检索（语义相似度）
        bm25_docs = self.bm25_search(query, candidate_k)             # BM25 关键词检索（精确匹配）

        # 标记每路来源（dual_level 内部会写 search_type 但不一定写 search_method）
        for d in dual_docs:
            d.metadata.setdefault("search_method", "dual_level")  # 标记来源（首次写入）
        for d in vector_docs:
            d.metadata["search_method"] = "vector"               # 标记来源（覆盖）
        # bm25_search 内部已写 search_method=bm25，无需重复标记

        # RRF 融合三路结果（去重 + 加权排名）
        final_docs = self._rrf_merge(
            ranked_lists=[
                ("dual_level", dual_docs),  # (source_name, ranked_docs) — source_name 用于 RRF 元信息
                ("vector", vector_docs),
                ("bm25", bm25_docs),
            ],
            top_k=top_k,
        )

        # 父文档回填（可选启用，仅 hybrid_traditional 路；不改排名，仅换上下文内容）
        if getattr(self.config, "enable_parent_doc_retrieval", False):
            final_docs = self._attach_parent_documents(final_docs)

        # 缓存通道统计（供 Web API 透传到前端展示三路召回贡献）
        self.last_hybrid_stats = {
            "candidates": {
                "dual_level": len(dual_docs),
                "vector": len(vector_docs),
                "bm25": len(bm25_docs),
            },
            "final": len(final_docs),
            "channels": ["dual_level", "vector", "bm25"],
        }

        logger.info(
            f"RRF 融合完成：dual={len(dual_docs)} vector={len(vector_docs)} "
            f"bm25={len(bm25_docs)} → 最终 {len(final_docs)} 个文档"
        )
        return final_docs

    def close(self):
        """关闭 Neo4j 资源连接。"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")