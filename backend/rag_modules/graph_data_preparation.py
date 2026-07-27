"""
图数据库数据准备模块

负责从 Neo4j 读取知识图谱，将菜谱、食材、烹饪步骤等节点转换为
LangChain Document 对象（page_content + metadata），供后续的 Milvus 向量化和检索模块使用。

核心流程：
    1. Neo4j 连接 → 加载所有 Recipe / Ingredient / CookingStep 节点
    2. 组装每道菜谱的完整文档（描述 + 食材列表 + 步骤列表）
    3. 按章节或固定长度分块（chunk），用于向量索引构建

该模块的输入源是 Neo4j 图数据库，输出是 Document 列表，
是整个 RAG pipeline 的数据入口。
"""

import logging
import json
import os
import pickle
import hashlib
from collections import defaultdict
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from neo4j import GraphDatabase
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """图节点数据结构：封装 Neo4j 中的单个节点。

    Attributes:
        node_id: 节点在图数据库中的唯一标识（nodeId）
        labels: 节点的标签列表（如 ["Recipe"]、["Ingredient"]）
        name: 节点名称（菜品名/食材名/步骤描述）
        properties: 节点的属性字典，包含 category、cuisineType 等业务字段
    """
    node_id: str
    labels: List[str]
    name: str
    properties: Dict[str, Any]


@dataclass
class GraphRelation:
    """图关系数据结构：封装图中两个节点之间的关系。

    Attributes:
        start_node_id: 源节点的 nodeId
        end_node_id: 目标节点的 nodeId
        relation_type: 关系类型（REQUIRES / CONTAINS_STEP / BELONGS_TO_CATEGORY）
        properties: 关系的属性字典
    """
    start_node_id: str
    end_node_id: str
    relation_type: str
    properties: Dict[str, Any]


class GraphDataPreparationModule:
    """图数据库数据准备模块 — 从 Neo4j 读取数据并转换为文档。

    这是整个 RAG pipeline 的数据入口：
        Neo4j → (load_graph_data) → GraphNode 列表
                              ↓
                      (build_recipe_documents) → Document 列表
                              ↓
                   (chunk_documents) → 分块后的 Document 列表

    每个 Document 的 page_content 是自然语言描述的完整菜谱（包含描述、食材、步骤），
    metadata 中存储 nodeId / recipe_name / category 等结构化字段，便于后续检索和展示。
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j",
                 docs_cache_dir: str = ""):
        """初始化图数据库连接。

        Args:
            uri: Neo4j 连接的 Bolt URI（如 bolt://localhost:7687）
            user: 认证用户名
            password: 认证密码
            database: Neo4j 数据库名称（默认 "neo4j"）
            docs_cache_dir: 菜谱文档缓存目录（空则用模块同级 .recipe_docs_cache/）
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.docs_cache_dir = docs_cache_dir
        self.driver = None
        self.documents: List[Document] = []   # 完整菜谱 Document 列表（分块前）
        self.chunks: List[Document] = []       # 分块后的 Document 列表
        self.recipes: List[GraphNode] = []     # Recipe 节点
        self.ingredients: List[GraphNode] = [] # Ingredient 节点
        self.cooking_steps: List[GraphNode] = []  # CookingStep 节点

        self._connect()

    def _connect(self):
        """建立 Neo4j 连接，并测试连通性。

        使用 GraphDatabase.driver() 创建持久化连接池，
        随后通过 RETURN 1 测试连通性。若失败则抛出异常中断初始化。
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                database=self.database,
            )
            logger.info(f"已连接到Neo4j数据库: {self.uri} (user={self.user}, database={self.database})")

            # 测试连接：执行简单的 RETURN 1 语句，验证认证和连通性。
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_result = result.single()
                if test_result:
                    logger.info("Neo4j连接测试成功")

        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise

    def close(self):
        """关闭数据库连接，释放资源。"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")

    def load_graph_data(self) -> Dict[str, Any]:
        """从 Neo4j 加载所有图数据（菜谱、食材、烹饪步骤节点）。

        使用 Cypher 查询按条件提取三种类型的节点，并将 Category
        BELONGS_TO_CATEGORY 关系中的分类信息合并到 Recipe 节点的 properties 中。

        Returns:
            包含各类型节点计数的字典：{'recipes': N, 'ingredients': M, 'cooking_steps': K}
        """
        logger.info("正在从Neo4j加载图数据...")

        with self.driver.session() as session:
            # ---------------------------------------------------------------
            # 1. 加载所有 Recipe 节点，同时 JOIN Category 关系获取分类信息
            # ---------------------------------------------------------------
            recipes_query = """
            MATCH (r:Recipe)
            WHERE r.nodeId >= '200000000'
            OPTIONAL MATCH (r)-[:BELONGS_TO_CATEGORY]->(c:Category)
            WITH r, collect(c.name) as categories
            RETURN r.nodeId as nodeId, labels(r) as labels, r.name as name,
                   properties(r) as originalProperties,
                   CASE WHEN size(categories) > 0
                        THEN categories[0]
                        ELSE COALESCE(r.category, '未知') END as mainCategory,
                   CASE WHEN size(categories) > 0
                        THEN categories
                        ELSE [COALESCE(r.category, '未知')] END as allCategories
            ORDER BY r.nodeId
            """

            result = session.run(recipes_query)
            self.recipes = []
            for record in result:
                # 合并原始属性和新的分类信息（mainCategory / allCategories）
                properties = dict(record["originalProperties"])
                properties["category"] = record["mainCategory"]
                properties["all_categories"] = record["allCategories"]

                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=properties
                )
                self.recipes.append(node)

            logger.info(f"加载了 {len(self.recipes)} 个菜谱节点")

            # ---------------------------------------------------------------
            # 2. 加载所有 Ingredient 节点（食材）
            # ---------------------------------------------------------------
            ingredients_query = """
            MATCH (i:Ingredient)
            WHERE i.nodeId >= '200000000'
            RETURN i.nodeId as nodeId, labels(i) as labels, i.name as name,
                   properties(i) as properties
            ORDER BY i.nodeId
            """

            result = session.run(ingredients_query)
            self.ingredients = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.ingredients.append(node)

            logger.info(f"加载了 {len(self.ingredients)} 个食材节点")

            # ---------------------------------------------------------------
            # 3. 加载所有 CookingStep 节点（烹饪步骤）
            # ---------------------------------------------------------------
            steps_query = """
            MATCH (s:CookingStep)
            WHERE s.nodeId >= '200000000'
            RETURN s.nodeId as nodeId, labels(s) as labels, s.name as name,
                   properties(s) as properties
            ORDER BY s.nodeId
            """

            result = session.run(steps_query)
            self.cooking_steps = []
            for record in result:
                node = GraphNode(
                    node_id=record["nodeId"],
                    labels=record["labels"],
                    name=record["name"],
                    properties=record["properties"]
                )
                self.cooking_steps.append(node)

            logger.info(f"加载了 {len(self.cooking_steps)} 个烹饪步骤节点")

        return {
            'recipes': len(self.recipes),
            'ingredients': len(self.ingredients),
            'cooking_steps': len(self.cooking_steps)
        }

    def build_recipe_documents(self, force_refresh: bool = False) -> List[Document]:
        """组装每道菜谱的完整文档，包含描述、食材列表和烹饪步骤。

        优化点：
            1. UNWIND 批量查询：把「每菜谱 2 次 Cypher」压成「全局 2 次」，
               消除 N+1 查询（N 个菜谱原本要 2N 次数据库往返）。
            2. 文档结果持久化：图数据未变时直接读磁盘缓存，跳过整个构建。

        Args:
            force_refresh: True 时强制重新构建（忽略缓存），用于 rebuild 场景

        Returns:
            完整菜谱 Document 列表（分块前）
        """
        logger.info("正在构建菜谱文档...")

        # 方案 3：优先读文档缓存（图数据未变则跳过整个构建，含 Neo4j 查询）
        fingerprint = self._compute_docs_fingerprint()
        if not force_refresh:
            cached = self._load_documents_cache(fingerprint)
            if cached is not None:
                self.documents = cached
                logger.info(f"从缓存加载 {len(cached)} 个菜谱文档（跳过 Neo4j 查询）")
                return cached

        documents = []
        recipe_ids = [r.node_id for r in self.recipes]

        # 方案 1：UNWIND 批量查询，2 次 Cypher 拿到全部菜谱的食材和步骤
        ingredients_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        steps_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        if recipe_ids:
            with self.driver.session() as session:
                # 1. 批量查所有菜谱的食材，按 recipe_id 分组
                ingredients_query = """
                UNWIND $recipe_ids AS rid
                MATCH (r:Recipe {nodeId: rid})-[req:REQUIRES]->(i:Ingredient)
                RETURN rid AS recipe_id, i.name AS name, i.category AS category,
                       req.amount AS amount, req.unit AS unit,
                       i.description AS description
                ORDER BY rid, i.name
                """
                for record in session.run(ingredients_query, {"recipe_ids": recipe_ids}):
                    ingredients_map[record["recipe_id"]].append(dict(record))

                # 2. 批量查所有菜谱的步骤，按 recipe_id 分组
                steps_query = """
                UNWIND $recipe_ids AS rid
                MATCH (r:Recipe {nodeId: rid})-[c:CONTAINS_STEP]->(s:CookingStep)
                RETURN rid AS recipe_id, s.name AS name, s.description AS description,
                       s.stepNumber AS stepNumber, s.methods AS methods,
                       s.tools AS tools, s.timeEstimate AS timeEstimate,
                       c.stepOrder AS stepOrder
                ORDER BY rid, COALESCE(c.stepOrder, s.stepNumber, 999)
                """
                for record in session.run(steps_query, {"recipe_ids": recipe_ids}):
                    steps_map[record["recipe_id"]].append(dict(record))

            # 3. 遍历菜谱，从内存 map 取数据组装文档（不再查库）
            for recipe in self.recipes:
                try:
                    recipe_id = recipe.node_id
                    recipe_name = recipe.name

                    # 组装食材文本（逻辑与原实现一致）
                    ingredients_info = []
                    for ing_record in ingredients_map.get(recipe_id, []):
                        amount = ing_record.get("amount", "")
                        unit = ing_record.get("unit", "")
                        ingredient_text = f"{ing_record['name']}"
                        if amount and unit:
                            ingredient_text += f"({amount}{unit})"
                        if ing_record.get("description"):
                            ingredient_text += f" - {ing_record['description']}"
                        ingredients_info.append(ingredient_text)

                    # 组装步骤文本（逻辑与原实现一致）
                    steps_info = []
                    for step_record in steps_map.get(recipe_id, []):
                        step_text = f"步骤: {step_record['name']}"
                        if step_record.get("description"):
                            step_text += f"\n描述: {step_record['description']}"
                        if step_record.get("methods"):
                            step_text += f"\n方法: {step_record['methods']}"
                        if step_record.get("tools"):
                            step_text += f"\n工具: {step_record['tools']}"
                        if step_record.get("timeEstimate"):
                            step_text += f"\n时间: {step_record['timeEstimate']}"
                        steps_info.append(step_text)

                    # 组装完整菜谱文档内容（自然语言描述）
                    content_parts = [f"# {recipe_name}"]

                    # 添加菜谱基本信息（描述、菜系、难度等）
                    if recipe.properties.get("description"):
                        content_parts.append(f"\n## 菜品描述\n{recipe.properties['description']}")

                    if recipe.properties.get("cuisineType"):
                        content_parts.append(f"\n菜系: {recipe.properties['cuisineType']}")

                    if recipe.properties.get("difficulty"):
                        content_parts.append(f"难度: {recipe.properties['difficulty']}星")

                    if recipe.properties.get("prepTime") or recipe.properties.get("cookTime"):
                        time_info = []
                        if recipe.properties.get("prepTime"):
                            time_info.append(f"准备时间: {recipe.properties['prepTime']}")
                        if recipe.properties.get("cookTime"):
                            time_info.append(f"烹饪时间: {recipe.properties['cookTime']}")
                        content_parts.append(f"\n时间信息: {', '.join(time_info)}")

                    if recipe.properties.get("servings"):
                        content_parts.append(f"份量: {recipe.properties['servings']}")

                    # 添加食材信息（带用量和描述）
                    if ingredients_info:
                        content_parts.append("\n## 所需食材")
                        for i, ingredient in enumerate(ingredients_info, 1):
                            content_parts.append(f"{i}. {ingredient}")

                    # 添加步骤信息（按顺序排列）
                    if steps_info:
                        content_parts.append("\n## 制作步骤")
                        for i, step in enumerate(steps_info, 1):
                            content_parts.append(f"\n### 第{i}步\n{step}")

                    # 添加标签信息
                    if recipe.properties.get("tags"):
                        content_parts.append(f"\n## 标签\n{recipe.properties['tags']}")

                    # 组合成最终内容
                    full_content = "\n".join(content_parts)

                    # 创建 Document 对象（page_content + metadata）
                    doc = Document(
                        page_content=full_content,
                        metadata={
                            "node_id": recipe_id,                  # 菜谱在 Neo4j 中的唯一标识
                            "recipe_name": recipe_name,              # 菜谱名称（展示用）
                            "node_type": "Recipe",                   # 节点类型标记
                            "category": recipe.properties.get("category", "未知"),
                            "cuisine_type": recipe.properties.get("cuisineType", "未知"),
                            "difficulty": recipe.properties.get("difficulty", 0),
                            "prep_time": recipe.properties.get("prepTime", ""),
                            "cook_time": recipe.properties.get("cookTime", ""),
                            "servings": recipe.properties.get("servings", ""),
                            "ingredients_count": len(ingredients_info),
                            "steps_count": len(steps_info),
                            "doc_type": "recipe",                    # 文档类型标记（供检索模块区分）
                            "content_length": len(full_content)      # 内容长度（供分块参考）
                        }
                    )

                    documents.append(doc)

                except Exception as e:
                    logger.warning(f"构建菜谱文档失败 {recipe_name} (ID: {recipe_id}): {e}")
                    continue

        self.documents = documents
        logger.info(f"成功构建 {len(documents)} 个菜谱文档")

        # 方案 3：构建结果写入缓存（供下次启动复用）
        self._save_documents_cache(fingerprint, documents)
        return documents

    # ========== 菜谱文档缓存（方案 3：持久化构建结果） ==========
    # 图数据未变时（指纹匹配），启动直接 load 文档，跳过 Neo4j 批量查询。

    def _get_docs_cache_dir(self) -> str:
        """获取菜谱文档缓存目录。

        优先使用构造时传入的 docs_cache_dir；未传时回退到模块同级 .recipe_docs_cache/。
        目录不存在时自动创建。
        """
        cache_dir = self.docs_cache_dir or ""
        if not cache_dir:
            cache_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), ".recipe_docs_cache"
            )
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _compute_docs_fingerprint(self) -> str:
        """计算菜谱文档指纹：基于 recipes 节点内容 + 食材/步骤节点规模。

        recipe 节点内容（含 properties）或节点规模变化，指纹即变，缓存自动失效。

        Note:
            若仅修改关系（增删食材/步骤）而 recipe 节点未变，指纹不变；
            此时需用 rebuild 或 clear_documents_cache() 强制刷新。
        """
        h = hashlib.md5()
        for r in self.recipes:
            h.update(f"{r.node_id}:{r.name}:".encode("utf-8"))
            props_str = json.dumps(r.properties, sort_keys=True, ensure_ascii=False)
            h.update(props_str.encode("utf-8"))
            h.update(b"\n")
        h.update(f"ingredients:{len(self.ingredients)}\n".encode("utf-8"))
        h.update(f"cooking_steps:{len(self.cooking_steps)}\n".encode("utf-8"))
        return h.hexdigest()

    def _load_documents_cache(self, fingerprint: str) -> Optional[List[Document]]:
        """加载缓存的菜谱文档（指纹不匹配或读取失败均视为未命中）。

        Args:
            fingerprint: 当前图数据的指纹

        Returns:
            Document 列表（命中时）；None（未命中或失败时）
        """
        cache_path = os.path.join(self._get_docs_cache_dir(), f"recipe_docs_{fingerprint[:16]}.pkl")
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            if data.get("fingerprint") != fingerprint:
                logger.info("菜谱文档缓存指纹不匹配，丢弃重建")
                return None
            logger.info(f"命中菜谱文档缓存: {cache_path}")
            return data.get("documents")
        except Exception as e:
            logger.warning(f"加载菜谱文档缓存失败，将重新构建: {e}")
            return None

    def _save_documents_cache(self, fingerprint: str, documents: List[Document]) -> None:
        """保存菜谱文档到磁盘缓存（失败不影响主流程）。"""
        cache_path = os.path.join(self._get_docs_cache_dir(), f"recipe_docs_{fingerprint[:16]}.pkl")
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "fingerprint": fingerprint,
                        "documents": documents,
                        "recipe_count": len(documents),
                        "method": "neo4j_batch_unwind",
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            logger.info(f"菜谱文档已缓存: {cache_path}")
        except Exception as e:
            logger.warning(f"保存菜谱文档缓存失败（不影响运行）: {e}")

    def clear_documents_cache(self) -> None:
        """清空菜谱文档缓存目录（用于 rebuild 场景强制重建）。"""
        cache_dir = self._get_docs_cache_dir()
        removed = 0
        try:
            for name in os.listdir(cache_dir):
                if name.endswith(".pkl"):
                    os.remove(os.path.join(cache_dir, name))
                    removed += 1
            logger.info(f"已清理菜谱文档缓存：删除 {removed} 个文件")
        except Exception as e:
            logger.warning(f"清理菜谱文档缓存失败: {e}")

    def chunk_documents(self, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        """对文档进行分块处理。

        两种分块策略（优先级从高到低）：
          1. **按章节分块**：以 `## ` 二级标题分割（优先保留语义完整性）
          2. **按长度强制分块**：当文档没有二级标题或单个章节过长时，
             以 chunk_size 为目标、chunk_overlap 为重叠量滑动切割。

        Args:
            chunk_size: 每个分块的目标字符数（默认 500）
            chunk_overlap: 相邻分块之间的重叠字符数（默认 50），用于避免切分打断上下文

        Returns:
            分块后的 Document 列表（每个 chunk 都携带 parent_id、chunk_index 等元信息）
        """
        logger.info(f"正在进行文档分块，块大小: {chunk_size}, 重叠: {chunk_overlap}")

        if not self.documents:
            raise ValueError("请先构建文档")

        chunks = []
        chunk_id = 0

        for doc in self.documents:
            content = doc.page_content

            # ------------------------------------------
            # 策略 A：内容较短，无需分块（整篇就是一个 chunk）
            # ------------------------------------------
            if len(content) <= chunk_size:
                chunk = Document(
                    page_content=content,
                    metadata={
                        **doc.metadata,
                        "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",  # 全局 chunk 编号
                        "parent_id": doc.metadata["node_id"],                         # 所属菜谱的 nodeId
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "chunk_size": len(content),
                        "doc_type": "chunk"
                    }
                )
                chunks.append(chunk)
                chunk_id += 1

            else:
                # ------------------------------------------
                # 策略 B：尝试按二级标题（## ）分块，如果段落数 <=1 则退化为按长度强制分块
                # ------------------------------------------
                sections = content.split('\n## ')

                if len(sections) <= 1:
                    # 没有二级标题，按长度强制分块（滑动窗口，保留 overlap）
                    total_chunks = (len(content) - 1) // (chunk_size - chunk_overlap) + 1

                    for i in range(total_chunks):
                        start = i * (chunk_size - chunk_overlap)
                        end = min(start + chunk_size, len(content))

                        chunk_content = content[start:end]

                        chunk = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                                "parent_id": doc.metadata["node_id"],
                                "chunk_index": i,
                                "total_chunks": total_chunks,
                                "chunk_size": len(chunk_content),
                                "doc_type": "chunk"
                            }
                        )
                        chunks.append(chunk)
                        chunk_id += 1

                else:
                    # 按二级标题分块（每段以 ## 开头，保留章节结构）
                    total_chunks = len(sections)
                    for i, section in enumerate(sections):
                        if i == 0:
                            # 第一个部分包含标题（如 "# 宫保鸡丁"），直接使用。
                            chunk_content = section
                        else:
                            # 其他部分添加章节标题，保持可读性（如 "## 所需食材\..."）
                            chunk_content = f"## {section}"

                        chunk = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                "chunk_id": f"{doc.metadata['node_id']}_chunk_{chunk_id}",
                                "parent_id": doc.metadata["node_id"],
                                "chunk_index": i,
                                "total_chunks": total_chunks,
                                "chunk_size": len(chunk_content),
                                "doc_type": "chunk",
                                "section_title": section.split('\n')[0] if i > 0 else "主标题"
                            }
                        )
                        chunks.append(chunk)
                        chunk_id += 1

        self.chunks = chunks
        logger.info(f"文档分块完成，共生成 {len(chunks)} 个块")
        return chunks

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息。

        Returns:
            包含以下信息的字典：
              - total_recipes / ingredients / cooking_steps: 节点计数
              - total_documents / total_chunks: 文档和分块计数
              - categories / cuisines / difficulties: 分类统计
              - avg_content_length / avg_chunk_size: 平均内容长度和分块大小
        """
        stats = {
            'total_recipes': len(self.recipes),
            'total_ingredients': len(self.ingredients),
            'total_cooking_steps': len(self.cooking_steps),
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks)
        }

        if self.documents:
            # 分类统计：按菜系、难度、菜品分类分别计数。
            categories = {}
            cuisines = {}
            difficulties = {}

            for doc in self.documents:
                category = doc.metadata.get('category', '未知')
                categories[category] = categories.get(category, 0) + 1

                cuisine = doc.metadata.get('cuisine_type', '未知')
                cuisines[cuisine] = cuisines.get(cuisine, 0) + 1

                difficulty = doc.metadata.get('difficulty', 0)
                difficulties[str(difficulty)] = difficulties.get(str(difficulty), 0) + 1

            stats.update({
                'categories': categories,
                'cuisines': cuisines,
                'difficulties': difficulties,
                'avg_content_length': sum(doc.metadata.get('content_length', 0) for doc in self.documents) / len(self.documents),
                'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
            })

        return stats

    # ========== 知识子图（供前端可视化） ==========

    @staticmethod
    def _primary_label(labels: List[str]) -> str:
        """从标签列表中取主标签（Recipe/Ingredient/CookingStep/Category 优先）。"""
        for lbl in ("Recipe", "Ingredient", "CookingStep", "Category"):
            if lbl in labels:
                return lbl
        return labels[0] if labels else "Unknown"

    @staticmethod
    def _summarize_properties(label: str, props: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """按节点类型精简属性，供前端 tooltip 展示，避免传输冗余字段。"""
        if not props:
            return {}
        key_map = {
            "Recipe": ("category", "cuisineType", "difficulty", "servings"),
            "Ingredient": ("category", "description"),
            "CookingStep": ("description", "methods", "tools", "timeEstimate"),
        }
        keys = key_map.get(label, ())
        return {k: props[k] for k in keys if props.get(k) is not None}

    def _add_neighbor(
        self,
        nodes: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]],
        *,
        neighbor_id: str,
        neighbor_labels: List[str],
        neighbor_name: Optional[str],
        neighbor_props: Optional[Dict[str, Any]],
        from_id: str,
        to_id: str,
        rel_type: str,
    ) -> None:
        """把邻居节点加入 nodes（去重），并追加一条 from_id->to_id 的边。

        边方向统一为 Recipe->邻居（REQUIRES/CONTAINS_STEP/BELONGS_TO_CATEGORY）。
        """
        if neighbor_id and neighbor_id not in nodes:
            label = self._primary_label(neighbor_labels)
            nodes[neighbor_id] = {
                "id": neighbor_id,
                "label": neighbor_name or neighbor_id,
                "type": label,
                "properties": self._summarize_properties(label, neighbor_props),
            }
        edges.append({"from": from_id, "to": to_id, "type": rel_type})

    def get_knowledge_subgraph(
        self, node_type: str, limit: int = 15, neighbor_limit: int = 6
    ) -> Dict[str, Any]:
        """返回某类节点的有界知识子图，供前端可视化。

        主节点取自内存列表（recipes/ingredients/cooking_steps）前 limit 个，
        元数据免查库；邻居与关系通过 Neo4j 查询，每主节点邻居数封顶 neighbor_limit，
        避免超大图（食材/步骤各近 3000）无法渲染。

        Args:
            node_type: recipes | ingredients | cooking_steps
            limit: 主节点数量上限（clamp 到 [1,50]）
            neighbor_limit: 每个主节点的邻居数量上限（clamp 到 [1,12]）

        Returns:
            {"nodes": [...], "edges": [...], "counts": {"primary": N, "total": M}}

        Raises:
            ValueError: node_type 非法
        """
        node_type = (node_type or "").lower()
        if node_type == "recipes":
            primary_nodes = self.recipes
        elif node_type == "ingredients":
            primary_nodes = self.ingredients
        elif node_type == "cooking_steps":
            primary_nodes = self.cooking_steps
        else:
            raise ValueError(
                f"未知节点类型: {node_type}（可选: recipes/ingredients/cooking_steps）"
            )

        limit = max(1, min(int(limit or 15), 50))
        neighbor_limit = max(1, min(int(neighbor_limit or 6), 12))
        primary_nodes = primary_nodes[:limit]
        if not primary_nodes:
            return {"nodes": [], "edges": [], "counts": {"primary": 0, "total": 0}}

        primary_ids = [n.node_id for n in primary_nodes]
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []
        neighbor_count: Dict[str, int] = {pid: 0 for pid in primary_ids}

        # 主节点（元数据来自内存）
        for n in primary_nodes:
            label = self._primary_label(n.labels)
            nodes[n.node_id] = {
                "id": n.node_id,
                "label": n.name or n.node_id,
                "type": label,
                "properties": self._summarize_properties(label, n.properties),
            }

        # 邻居 + 关系（查 Neo4j），每主节点邻居封顶 neighbor_limit
        with self.driver.session() as session:
            if node_type == "recipes":
                # Recipe -> Ingredient / CookingStep / Category
                # Category 节点没有 nodeId，用 'cat:'+name 作 id（与数字 nodeId 不冲突）
                # 按关系类型分别封顶：Category 仅取 1 个，其余每类取 neighbor_limit 个，
                # 保证每个菜谱同时出现食材/步骤/分类三类邻居，而非被字母序挤掉。
                query = """
                UNWIND $ids AS rid
                MATCH (r:Recipe {nodeId: rid})-[rel]->(n)
                WHERE n:Ingredient OR n:CookingStep OR n:Category
                RETURN rid AS rid, type(rel) AS rel_type,
                       COALESCE(n.nodeId, 'cat:' + n.name) AS nid,
                       labels(n) AS nlabels, n.name AS nname,
                       properties(n) AS nprops
                ORDER BY rid, rel_type, nname
                """
                per_type_count: Dict[tuple, int] = {}
                type_cap = {"BELONGS_TO_CATEGORY": 1}
                for rec in session.run(query, {"ids": primary_ids}):
                    rid = rec["rid"]
                    rel_type = rec["rel_type"]
                    key = (rid, rel_type)
                    cap = type_cap.get(rel_type, neighbor_limit)
                    if per_type_count.get(key, 0) >= cap:
                        continue
                    per_type_count[key] = per_type_count.get(key, 0) + 1
                    self._add_neighbor(
                        nodes, edges,
                        neighbor_id=rec["nid"], neighbor_labels=rec["nlabels"],
                        neighbor_name=rec["nname"], neighbor_props=rec["nprops"],
                        from_id=rid, to_id=rec["nid"], rel_type=rel_type,
                    )

            elif node_type == "ingredients":
                # Ingredient <- REQUIRES - Recipe  =>  边 Recipe -> Ingredient
                query = """
                UNWIND $ids AS iid
                MATCH (i:Ingredient {nodeId: iid})<-[rel:REQUIRES]-(r:Recipe)
                RETURN iid AS iid, r.nodeId AS rid, r.name AS rname, properties(r) AS rprops
                ORDER BY iid, rname
                """
                for rec in session.run(query, {"ids": primary_ids}):
                    iid = rec["iid"]
                    if neighbor_count.get(iid, 0) >= neighbor_limit:
                        continue
                    neighbor_count[iid] += 1
                    self._add_neighbor(
                        nodes, edges,
                        neighbor_id=rec["rid"], neighbor_labels=["Recipe"],
                        neighbor_name=rec["rname"], neighbor_props=rec["rprops"],
                        from_id=rec["rid"], to_id=iid, rel_type="REQUIRES",
                    )

            else:  # cooking_steps
                # CookingStep <- CONTAINS_STEP - Recipe  =>  边 Recipe -> CookingStep
                query = """
                UNWIND $ids AS sid
                MATCH (s:CookingStep {nodeId: sid})<-[rel:CONTAINS_STEP]-(r:Recipe)
                RETURN sid AS sid, r.nodeId AS rid, r.name AS rname, properties(r) AS rprops
                ORDER BY sid, rname
                """
                for rec in session.run(query, {"ids": primary_ids}):
                    sid = rec["sid"]
                    if neighbor_count.get(sid, 0) >= neighbor_limit:
                        continue
                    neighbor_count[sid] += 1
                    self._add_neighbor(
                        nodes, edges,
                        neighbor_id=rec["rid"], neighbor_labels=["Recipe"],
                        neighbor_name=rec["rname"], neighbor_props=rec["rprops"],
                        from_id=rec["rid"], to_id=sid, rel_type="CONTAINS_STEP",
                    )

        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            "counts": {"primary": len(primary_ids), "total": len(nodes)},
        }

    def __del__(self):
        """析构函数：确保关闭数据库连接（防止资源泄漏）。"""
        self.close()