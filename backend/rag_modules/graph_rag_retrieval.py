"""
真正的图 RAG 检索模块 - 基于 Neo4j 的深度知识推理

传统 RAG 仅做文本相似度匹配，本模块利用 Neo4j 图数据库的拓扑结构优势，
实现多跳推理（multi-hop reasoning）和知识子图提取。

核心能力：
    1. 查询意图理解：将自然语言转换为图查询模式（entity_relation / multi_hop 等）
    2. 多跳图遍历：深度关系探索（1-3 跳），发现隐含知识关联
    3. 子图提取：围绕核心实体的局部知识网络（含图谱指标计算）
    4. 图结构推理：基于拓扑的因果 / 组成 / 相似推理
    5. 动态查询规划：根据查询复杂度自适应调整遍历策略

与 HybridRetrievalModule 的区别：
    - hybrid_retrieval：基于预建索引的「键值对匹配 + 向量相似度」，适合简单信息查找
    - graph_rag：基于图拓扑的「查询规划 + 多跳遍历」，适合复杂关系推理

外部依赖：
    - Neo4j GraphDatabase.driver() 提供图遍历能力
"""

import json
import logging
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """查询类型枚举：将用户的自然语言问题映射到图数据库的查询模式。

    Attributes:
        ENTITY_RELATION: 直接实体关系（"鸡肉和胡萝卜能一起做菜吗？"）
        MULTI_HOP: 多跳推理（"鸡肉配什么蔬菜？" -> 菜品->食材->蔬菜）
        SUBGRAPH: 完整子图（"川菜有什么特色？" -> 需要相关完整知识网络）
        PATH_FINDING: 路径查找（"从食材到成品菜的制作路径"）
        CLUSTERING: 聚类相似性（"和宫保鸡丁类似的菜有哪些？"）
    """
    ENTITY_RELATION = "entity_relation"  # 直接实体关系（一跳）
    MULTI_HOP = "multi_hop"             # 多跳推理（2-3 跳）
    SUBGRAPH = "subgraph"               # 完整子图（局部知识网络）
    PATH_FINDING = "path_finding"       # 路径查找（最短路径）
    CLUSTERING = "clustering"           # 聚类相似性


@dataclass
class GraphQuery:
    """图查询结构：将用户的自然语言问题转换为可执行的图数据库查询计划。

    Attributes:
        query_type: 查询类型（entity_relation / multi_hop / subgraph 等）
        source_entities: 查询的起点实体（如"川菜""鸡胸肉"等图内可能存在的具体节点）
        target_entities: 查询的目标实体（仅 path_finding / multi_hop 时使用，如"蔬菜"）
        relation_types: 优先考虑的关系类型（如 ["REQUIRES", "BELONGS_TO_CATEGORY"]）
        max_depth: 图遍历最大跳数（1-3，控制推理深度）
        max_nodes: 子图提取的最大节点数（默认 50，防止内存溢出）
        constraints: 可选的属性级约束（如健康限制"糖尿病"/时间限制"30分钟"）
    """
    query_type: QueryType
    source_entities: List[str]
    target_entities: List[str] = None       # 目标实体（path_finding / multi_hop 时使用）
    relation_types: List[str] = None        # 优先考虑的关系类型
    max_depth: int = 2                      # 图遍历最大跳数（1-3）
    max_nodes: int = 50                     # 子图提取的最大节点数
    constraints: Dict[str, Any] = None      # 可选的属性级约束（健康/时间等）


@dataclass
class GraphPath:
    """图路径结构：封装多跳遍历的结果（节点序列 + 关系序列）。

    Attributes:
        nodes: 路径上的所有节点信息（id, name, labels, properties）
        relationships: 路径上所有的关系（type, properties）
        path_length: 路径长度（跳数 = 边数）
        relevance_score: 路径相关性得分（综合跳数、节点度数、关系类型匹配）
        path_type: 路径类型标记（"multi_hop"）
    """
    nodes: List[Dict[str, Any]]           # 路径上的节点信息列表
    relationships: List[Dict[str, Any]]    # 路径上的关系信息列表
    path_length: int                       # 路径长度（跳数）
    relevance_score: float                 # 相关性得分
    path_type: str                         # 路径类型标记


@dataclass
class KnowledgeSubgraph:
    """知识子图结构：封装子图提取的结果（中心节点 + 连通节点 + 关系）。

    Attributes:
        central_nodes: 子图的核心实体（如"川菜"对应的节点）
        connected_nodes: 与核心实体关联的节点列表
        relationships: 子图中的所有关系（边）
        graph_metrics: 图指标（密度、节点数等，用于后续分析）
        reasoning_chains: 基于图结构的推理链列表（因果/组成/相似关系）
    """
    central_nodes: List[Dict[str, Any]]    # 子图的核心实体
    connected_nodes: List[Dict[str, Any]]  # 与核心实体关联的节点列表
    relationships: List[Dict[str, Any]]    # 子图中的所有关系（边）
    graph_metrics: Dict[str, float]        # 图指标（密度、节点数等）
    reasoning_chains: List[List[str]]      # 基于图结构的推理链列表


class GraphRAGRetrieval:
    """真正的图 RAG 检索系统

    核心特点：
        1. 查询意图理解：识别图查询模式（entity_relation / multi_hop / subgraph 等）
        2. 多跳图遍历：深度关系探索（基于 Cypher 路径查询）
        3. 子图提取：相关知识网络（含密度等图谱指标计算）
        4. 图结构推理：基于拓扑的因果 / 组成 / 相似推理链
        5. 动态查询规划：自适应遍历策略（按复杂度选择 1/2/3 跳）

    检索流程：
        query
          ↓ (understand_graph_query - LLM 意图理解)
        GraphQuery
          ↓ (根据 query_type 分派)
        ┌─────────────┬──────────────┬─────────────┐
        │ MULTI_HOP    │ SUBGRAPH     │ ENTITY_REL │
        ↓              ↓              ↓
    multi_hop_traversal  extract_knowledge_subgraph  multi_hop_traversal
          ↓              ↓ (graph_structure_reasoning)
        GraphPath       KnowledgeSubgraph
          ↓              ↓
        _paths_to_documents / _subgraph_to_documents
          ↓
        Document 列表（metadata 含 search_type="graph_path" / "knowledge_subgraph"）

    Public API:
        - graph_rag_search(query, top_k)  -> 主搜索接口，整合所有图 RAG 能力
        - explain_routing_decision(...)   -> 解释路由决策（供调试）

    Note:
        本模块依赖 LLM 进行查询意图理解（understand_graph_query），
        将自然语言问题映射到 GraphQuery 结构后，再用 Cypher 执行图遍历。
    """

    def __init__(self, config, llm_client):
        self.config = config
        self.llm_client = llm_client
        self.driver = None  # Neo4j 数据库连接（在 initialize() 中创建）

        # 图结构缓存：预热阶段构建，加速后续查询
        self.entity_cache = {}      # node_id -> {labels, name, category, degree}
        self.relation_cache = {}    # rel_type -> frequency
        self.subgraph_cache = {}    # 预留：子图缓存（当前未使用）

        # 最近一次图查询规划（dict | None）。
        # graph_rag_search 中由 understand_graph_query 的结果填充，供 Web API
        # 透传到前端展示「走了哪些关系、多少跳、源/目标实体」。
        # 每次进入 graph_rag_search 先置 None，保证非图查询不会读到陈旧值。
        self.last_query_plan = None

    def initialize(self):
        """初始化图 RAG 检索系统：连接 Neo4j + 预热图索引。

        初始化步骤：
            1. 连接 Neo4j 并测试连通性
            2. 预热实体和关系索引（_build_graph_index）：构建本地缓存加速查询
        """
        logger.info("初始化图RAG检索系统...")

        # 1. 连接 Neo4j
        try:
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            # 测试连接：执行 RETURN 1 验证认证和连通性
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j连接成功")
        except Exception as e:
            logger.error(f"Neo4j连接失败: {e}")
            return

        # 2. 预热：构建实体和关系索引（本地缓存，加速后续图查询）
        self._build_graph_index()

    def _build_graph_index(self):
        """构建图索引以加速查询（预热阶段构建本地缓存）。

        将 Neo4j 中的高频实体（按度数排序，Top 1000）和所有关系类型
        预加载到本地缓存，避免每次查询都需访问数据库。

        缓存内容：
            - entity_cache: node_id -> {labels, name, category, degree}
            - relation_cache: rel_type -> frequency（关系类型频次）
        """
        logger.info("构建图结构索引...")

        try:
            with self.driver.session() as session:
                # 构建实体索引 - 按 degree（度数）降序取 Top 1000，优先缓存高频节点
                entity_query = """
                MATCH (n)
                WHERE n.nodeId IS NOT NULL
                WITH n, COUNT { (n)--() } as degree
                RETURN labels(n) as node_labels, n.nodeId as node_id,
                       n.name as name, n.category as category, degree
                ORDER BY degree DESC
                LIMIT 1000
                """

                result = session.run(entity_query)
                for record in result:
                    node_id = record["node_id"]
                    self.entity_cache[node_id] = {
                        "labels": record["node_labels"],
                        "name": record["name"],
                        "category": record["category"],
                        "degree": record["degree"]  # 度数：衡量节点重要性的指标
                    }

                # 构建关系类型索引：统计每种关系类型的出现频次
                relation_query = """
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as frequency
                ORDER BY frequency DESC
                """

                result = session.run(relation_query)
                for record in result:
                    rel_type = record["rel_type"]
                    self.relation_cache[rel_type] = record["frequency"]

                logger.info(f"索引构建完成: {len(self.entity_cache)}个实体, {len(self.relation_cache)}个关系类型")

        except Exception as e:
            logger.error(f"构建图索引失败: {e}")

    def understand_graph_query(self, query: str) -> GraphQuery:
        """理解查询的图结构意图：将自然语言问题映射到 GraphQuery。

        这是图 RAG 的核心能力 - 通过 LLM 将用户问题转换为可执行的图查询计划：
            1. 识别查询类型（entity_relation / multi_hop / subgraph 等）
            2. 提取源实体（图内可能存在的具体节点，如"川菜""鸡肉"）
            3. 提取目标实体（仅 path_finding / multi_hop 时使用）
            4. 选择优先考虑的关系类型
            5. 设置遍历深度（1-3 跳）
            6. 提取属性级约束（健康/时间等非图结构过滤条件）

        Args:
            query: 用户的自然语言查询（如"鸡肉配什么蔬菜好？"）

        Returns:
            GraphQuery 对象（LLM 失败时降级为默认 subgraph 查询）
        """
        prompt = f"""
        作为图数据库专家，分析以下查询的图结构意图，并将自然语言问题映射到**已有图结构**上。

        已知图中大致有以下节点和关系：
        - 节点类型：
          - Recipe：菜谱节点，包含 name、description、cuisineType（如"川菜"）、category、tags、prepTime、cookTime 等属性
          - Ingredient：食材节点，包含 name、category（如"蔬菜"、"蛋白质" 等）
          - Category：菜品分类（如"川菜"、"家常菜"、"素菜"）
          - CookingStep：烹饪步骤
        - 主要关系：
          - (Recipe)-[:REQUIRES]->(Ingredient)
          - (Recipe)-[:BELONGS_TO_CATEGORY]->(Category)
          - (Recipe)-[:CONTAINS_STEP]->(CookingStep)

        请根据上述图结构分析下面的查询：

        查询：{query}

        请识别：
        1. 查询类型：
           - entity_relation: 询问实体间的直接关系（如：鸡肉和胡萝卜能一起做菜吗？）
           - multi_hop: 需要多跳推理（如：鸡肉配什么蔬菜？需要：鸡肉->菜品->食材->蔬菜）
           - subgraph: 需要完整子图（如：川菜有什么特色？需要川菜相关的完整知识网络）
           - path_finding: 路径查找（如：从食材到成品菜的制作路径）
           - clustering: 聚类相似性（如：和宫保鸡丁类似的菜有哪些？）

        2. source_entities：
           - 只包含在图中**很有可能有对应节点**的具体实体名称
           - 优先选择：菜系（如"川菜"）、具体菜名（如"宫保鸡丁"）、食材名（如"鸡肉"、"豆腐"）
           - 不要把抽象概念或约束（如"糖尿病饮食限制"、"具体川菜菜品"、"健康饮食"、"30分钟内"）放进 source_entities

        3. target_entities：
           - 只在确实需要限制「路径终点」时填写
           - 同样只能使用可能出现在 Recipe / Ingredient / Category 节点上的名称（如"蔬菜"、"素菜"、具体菜名）
           - 如果不确定目标实体怎么映射到图中，请返回空列表 []

        4. relation_types：本次推理中希望优先考虑的关系类型列表
           - 例如：["REQUIRES", "BELONGS_TO_CATEGORY"]

        5. max_depth：建议的图遍历深度（1-3 之间的整数）

        6. constraints：可选的**属性级约束**，用于表达图结构之外的过滤条件，例如：
           - 健康/饮食限制（如"糖尿病"、"低糖"）
           - 时间限制（如"30分钟内"）
           - 口味偏好（如"清淡"、"少油"）
           用一个字典描述，例如：
           {{
             "health": ["糖尿病", "低糖"],
             "time": {{"max_minutes": 30}},
             "style": ["川菜"]
           }}

        示例1：
        查询："鸡肉配什么蔬菜好？"
        期望分析：这是 multi_hop 查询，需要通过"鸡肉->使用鸡肉的菜品->这些菜品使用的蔬菜"的路径推理。

        返回JSON示例：
        {{
          "query_type": "multi_hop",
          "source_entities": ["鸡肉"],
          "target_entities": ["蔬菜"],
          "relation_types": ["REQUIRES", "BELONGS_TO_CATEGORY"],
          "max_depth": 3,
          "constraints": {{}}
        }}

        示例2：
        查询："适合糖尿病人吃的低糖川菜有哪些，并且制作时间不超过30分钟？"
        期望分析：
          - 图中可以直接对应的实体：主要是菜系 "川菜"
          - 糖尿病/低糖/30分钟 属于属性级约束，不能当作节点
          - 可以使用 subgraph 或 multi_hop，以 "川菜" 为核心实体，结合属性约束做后续过滤

        返回JSON示例：
        {{
          "query_type": "subgraph",
          "source_entities": ["川菜"],
          "target_entities": [],
          "relation_types": ["BELONGS_TO_CATEGORY", "REQUIRES"],
          "max_depth": 2,
          "constraints": {{
            "health": ["糖尿病", "低糖"],
            "time": {{"max_minutes": 30}}
          }}
        }}

        请严格返回一个合法的 JSON 对象，不要包含任何多余的说明文字。
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # 低温度确保输出格式稳定
                max_tokens=1000
            )

            result = json.loads(response.choices[0].message.content.strip())

            # 将 LLM 返回的 JSON 转换为 GraphQuery 对象
            return GraphQuery(
                query_type=QueryType(result.get("query_type", "subgraph")),  # 默认 subgraph
                source_entities=result.get("source_entities", []),
                target_entities=result.get("target_entities", []),
                relation_types=result.get("relation_types", []),
                max_depth=result.get("max_depth", 2),
                max_nodes=50
            )

        except Exception as e:
            logger.error(f"查询意图理解失败: {e}")
            # 降级方案：默认子图查询（以原 query 作为 source_entity）
            return GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=2
            )

    def multi_hop_traversal(self, graph_query: GraphQuery) -> List[GraphPath]:
        """多跳图遍历：图 RAG 的核心优势，通过图结构发现隐含的知识关联。

        根据 query_type 选择不同的遍历策略：
            - MULTI_HOP: 执行多跳路径查询，动态拼接目标过滤条件
            - ENTITY_RELATION: 调用 _find_entity_relations（当前为占位实现）
            - PATH_FINDING: 调用 _find_shortest_paths（当前为占位实现）

        路径评分公式（用于 relevance_score）：
            relevance = (1.0 / path_len)                           # 短路径得分高
                      + (avg_degree / 10.0)                        # 高度数节点得分高
                      + (0.3 if relation_type 匹配 else 0.0)       # 关系类型匹配加分

        Args:
            graph_query: 图查询计划（包含 source_entities / target_entities / max_depth 等）

        Returns:
            GraphPath 列表（按 relevance_score 降序，最多 20 条）
        """
        logger.info(f"执行多跳遍历: {graph_query.source_entities} -> {graph_query.target_entities}")

        paths = []

        if not self.driver:
            logger.error("Neo4j连接未建立")
            return paths

        try:
            with self.driver.session() as session:
                source_entities = graph_query.source_entities
                target_keywords = graph_query.target_entities or []
                max_depth = graph_query.max_depth

                # 根据 query_type 选择不同的遍历策略
                if graph_query.query_type == QueryType.MULTI_HOP:
                    # 根据是否有目标关键词动态拼接过滤条件
                    target_filter_clause = ""
                    if target_keywords:
                        # 目标过滤：匹配 target 节点的 name 或 category 包含目标关键词
                        target_filter_clause = """
                    AND ANY(kw IN $target_keywords WHERE
                        (target.name IS NOT NULL AND (toString(target.name) CONTAINS kw OR kw CONTAINS toString(target.name))) OR
                        (target.category IS NOT NULL AND (toString(target.category) CONTAINS kw OR kw CONTAINS toString(target.category)))
                    )"""

                    # 多跳路径查询：使用 Cypher 的变长路径语法 (source)-[*1..N]-(target)
                    cypher_query = f"""
                    // 多跳推理查询
                    UNWIND $source_entities as source_name
                    MATCH (source)
                    WHERE source.name CONTAINS source_name OR source.nodeId = source_name

                    // 执行多跳遍历（1 到 max_depth 跳）
                    MATCH path = (source)-[*1..{max_depth}]-(target)
                    WHERE NOT source = target{target_filter_clause}

                    // 计算路径相关性
                    WITH path, source, target,
                         length(path) as path_len,
                         relationships(path) as rels,
                         nodes(path) as path_nodes

                    // 路径评分：短路径 + 高度数节点 + 关系类型匹配
                    WITH path, source, target, path_len, rels, path_nodes,
                         (1.0 / path_len) +                                                          -- 短路径得分高
                         (REDUCE(s = 0.0, n IN path_nodes | s + COUNT {{ (n)--() }}) / 10.0 / size(path_nodes)) +  -- 高度数节点得分高
                         (CASE WHEN ANY(r IN rels WHERE type(r) IN $relation_types) THEN 0.3 ELSE 0.0 END) as relevance  -- 关系类型匹配加分

                    ORDER BY relevance DESC
                    LIMIT 20

                    RETURN path, source, target, path_len, rels, path_nodes, relevance
                    """

                    params = {
                        "source_entities": source_entities,
                        "relation_types": graph_query.relation_types or []
                    }
                    if target_keywords:
                        params["target_keywords"] = target_keywords

                    result = session.run(cypher_query, params)

                    for record in result:
                        path_data = self._parse_neo4j_path(record)
                        if path_data:
                            paths.append(path_data)

                elif graph_query.query_type == QueryType.ENTITY_RELATION:
                    # 实体间关系查询（一跳）
                    paths.extend(self._find_entity_relations(graph_query, session))

                elif graph_query.query_type == QueryType.PATH_FINDING:
                    # 最短路径查找
                    paths.extend(self._find_shortest_paths(graph_query, session))

        except Exception as e:
            logger.error(f"多跳遍历失败: {e}")

        logger.info(f"多跳遍历完成，找到 {len(paths)} 条路径")
        return paths

    def extract_knowledge_subgraph(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """提取知识子图：获取核心实体相关的完整知识网络。

        体现图 RAG 的整体性思维 - 不仅返回单条路径，而是返回围绕核心实体的
        完整局部子图（含节点、关系和图谱指标）。

        子图提取策略：
            1. 找到 source_entities 对应的核心节点
            2. 获取 max_depth 跳内的所有邻居节点和关系
            3. 计算图谱指标：节点数、关系数、密度（density = 边数 / 最大可能边数）

        Args:
            graph_query: 图查询计划（source_entities 作为子图核心）

        Returns:
            KnowledgeSubgraph 对象（失败时降级为空子图）
        """
        logger.info(f"提取知识子图: {graph_query.source_entities}")

        if not self.driver:
            logger.error("Neo4j连接未建立")
            return self._fallback_subgraph_extraction(graph_query)

        try:
            with self.driver.session() as session:
                # 简化的子图提取（不依赖 APOC 插件，使用原生 Cypher）
                cypher_query = f"""
                // 找到源实体（核心节点）
                UNWIND $source_entities as entity_name
                MATCH (source)
                WHERE source.name CONTAINS entity_name
                   OR source.nodeId = entity_name

                // 获取指定深度的邻居（1 到 max_depth 跳）
                MATCH (source)-[r*1..{graph_query.max_depth}]-(neighbor)
                WITH source, collect(DISTINCT neighbor) as neighbors,
                     collect(DISTINCT r) as relationships
                WHERE size(neighbors) <= $max_nodes

                // 计算图指标：节点数、关系数、密度
                WITH source, neighbors, relationships,
                     size(neighbors) as node_count,
                     size(relationships) as rel_count

                RETURN
                    source,
                    neighbors[0..{graph_query.max_nodes}] as nodes,
                    relationships[0..{graph_query.max_nodes}] as rels,
                    {{
                        node_count: node_count,
                        relationship_count: rel_count,
                        density: CASE WHEN node_count > 1 THEN toFloat(rel_count) / (node_count * (node_count - 1) / 2) ELSE 0.0 END
                    }} as metrics
                """

                result = session.run(cypher_query, {
                    "source_entities": graph_query.source_entities,
                    "max_nodes": graph_query.max_nodes
                })

                record = result.single()
                if record:
                    return self._build_knowledge_subgraph(record)

        except Exception as e:
            logger.error(f"子图提取失败: {e}")

        # 降级方案：返回空子图（避免影响主流程）
        return self._fallback_subgraph_extraction(graph_query)

    def graph_structure_reasoning(self, subgraph: KnowledgeSubgraph, query: str) -> List[str]:
        """基于图结构的推理：不仅检索信息，还能进行逻辑推理。

        推理流程：
            1. 识别推理模式（因果 / 组成 / 相似关系）
            2. 为每种模式构建推理链
            3. 验证推理链的可信度（当前为占位实现，返回前 3 条）

        Args:
            subgraph: 已提取的知识子图
            query: 原始用户查询（用于推理链验证）

        Returns:
            推理链列表（当前为占位实现，最多 3 条）
        """
        reasoning_chains = []

        try:
            # 1. 识别推理模式（当前为固定模式：因果 / 组成 / 相似）
            reasoning_patterns = self._identify_reasoning_patterns(subgraph)

            # 2. 为每种模式构建推理链
            for pattern in reasoning_patterns:
                chain = self._build_reasoning_chain(pattern, subgraph)
                if chain:
                    reasoning_chains.append(chain)

            # 3. 验证推理链的可信度（当前为占位实现，返回前 3 条）
            validated_chains = self._validate_reasoning_chains(reasoning_chains, query)

            logger.info(f"图结构推理完成，生成 {len(validated_chains)} 条推理链")
            return validated_chains

        except Exception as e:
            logger.error(f"图结构推理失败: {e}")
            return []

    def adaptive_query_planning(self, query: str) -> List[GraphQuery]:
        """自适应查询规划：根据查询复杂度动态调整遍历策略。

        复杂度分级：
            - < 0.3（简单）：ENTITY_RELATION，1 跳，最多 20 节点
            - 0.3-0.7（中等）：MULTI_HOP，2 跳，最多 50 节点
            - > 0.7（复杂）：SUBGRAPH + MULTI_HOP，3 跳，最多 100 节点

        Args:
            query: 用户的自然语言查询

        Returns:
            GraphQuery 计划列表（简单/中等返回 1 个，复杂返回 2 个并行计划）
        """
        # 分析查询复杂度（基于关键词匹配）
        complexity_score = self._analyze_query_complexity(query)

        query_plans = []

        if complexity_score < 0.3:
            # 简单查询：直接邻居查询（一跳）
            plan = GraphQuery(
                query_type=QueryType.ENTITY_RELATION,
                source_entities=[query],
                max_depth=1,
                max_nodes=20
            )
            query_plans.append(plan)

        elif complexity_score < 0.7:
            # 中等复杂度：多跳查询（2 跳）
            plan = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=[query],
                max_depth=2,
                max_nodes=50
            )
            query_plans.append(plan)

        else:
            # 复杂查询：子图提取 + 多跳推理（并行两个计划）
            plan1 = GraphQuery(
                query_type=QueryType.SUBGRAPH,
                source_entities=[query],
                max_depth=3,
                max_nodes=100
            )
            plan2 = GraphQuery(
                query_type=QueryType.MULTI_HOP,
                source_entities=[query],
                max_depth=3,
                max_nodes=50
            )
            query_plans.extend([plan1, plan2])

        return query_plans

    def graph_rag_search(self, query: str, top_k: int = 5) -> List[Document]:
        """图 RAG 主搜索接口：整合所有图 RAG 能力。

        执行流程：
            1. 查询意图理解（understand_graph_query）-> GraphQuery
            2. 根据 query_type 分派到不同策略：
               - MULTI_HOP / PATH_FINDING: multi_hop_traversal -> _paths_to_documents
               - SUBGRAPH / CLUSTERING: extract_knowledge_subgraph + graph_structure_reasoning
                 -> _subgraph_to_documents
               - ENTITY_RELATION: multi_hop_traversal（视为少量跳的路径查询）
            3. 基于图结构相关性排序（_rank_by_graph_relevance）

        Args:
            query: 用户的自然语言查询
            top_k: 返回结果数量上限

        Returns:
            Document 列表（metadata 含 search_type="graph_path" / "knowledge_subgraph"）
        """
        logger.info(f"开始图RAG检索: {query}")

        if not self.driver:
            logger.warning("Neo4j连接未建立，返回空结果")
            return []

        # 每次进入先清空，保证非图查询/失败时不会读到上一次的陈旧规划
        self.last_query_plan = None

        # 1. 查询意图理解：将自然语言转换为 GraphQuery
        graph_query = self.understand_graph_query(query)
        logger.info(f"查询类型: {graph_query.query_type.value}")

        # 缓存查询规划（供 Web API 透传到前端展示推理路径意图）
        self.last_query_plan = {
            "query_type": graph_query.query_type.value,
            "source_entities": list(graph_query.source_entities or []),
            "target_entities": list(graph_query.target_entities or []),
            "relation_types": list(graph_query.relation_types or []),
            "max_depth": graph_query.max_depth,
        }

        results = []

        try:
            # 2. 根据 query_type 执行不同策略
            if graph_query.query_type in [QueryType.MULTI_HOP, QueryType.PATH_FINDING]:
                # 多跳遍历 / 路径查找：执行路径查询并转换为 Document
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))

            elif graph_query.query_type in [QueryType.SUBGRAPH, QueryType.CLUSTERING]:
                # 子图提取 / 聚类查询：都视为"围绕核心实体的局部知识网络"
                subgraph = self.extract_knowledge_subgraph(graph_query)

                # 图结构推理：基于子图拓扑生成推理链
                reasoning_chains = self.graph_structure_reasoning(subgraph, query)

                results.extend(self._subgraph_to_documents(subgraph, reasoning_chains, query))

            elif graph_query.query_type == QueryType.ENTITY_RELATION:
                # 实体关系查询（可以视为一跳 / 少量跳的路径查询）
                paths = self.multi_hop_traversal(graph_query)
                results.extend(self._paths_to_documents(paths, query))

            # 3. 图结构相关性排序（按 relevance_score 降序）
            results = self._rank_by_graph_relevance(results, query)

            logger.info(f"图RAG检索完成，返回 {len(results[:top_k])} 个结果")
            return results[:top_k]

        except Exception as e:
            logger.error(f"图RAG检索失败: {e}")
            return []

    # ========== 辅助方法 ==========

    def _parse_neo4j_path(self, record) -> Optional[GraphPath]:
        """解析 Neo4j 路径记录为 GraphPath 对象。

        将 Cypher 返回的 path_nodes 和 rels 转换为结构化的节点/关系列表。

        Args:
            record: Neo4j 查询记录（包含 path_nodes, rels, path_len, relevance）

        Returns:
            GraphPath 对象（解析失败时返回 None）
        """
        try:
            # 解析路径上的所有节点
            path_nodes = []
            for node in record["path_nodes"]:
                path_nodes.append({
                    "id": node.get("nodeId", ""),
                    "name": node.get("name", ""),
                    "labels": list(node.labels),
                    "properties": dict(node)
                })

            # 解析路径上的所有关系
            relationships = []
            for rel in record["rels"]:
                relationships.append({
                    "type": rel.type,
                    "properties": dict(rel)
                })

            return GraphPath(
                nodes=path_nodes,
                relationships=relationships,
                path_length=record["path_len"],
                relevance_score=record["relevance"],
                path_type="multi_hop"
            )

        except Exception as e:
            logger.error(f"路径解析失败: {e}")
            return None

    def _build_knowledge_subgraph(self, record) -> KnowledgeSubgraph:
        """构建知识子图对象（从 Neo4j 记录转换为 KnowledgeSubgraph）。

        Args:
            record: Neo4j 查询记录（包含 source, nodes, rels, metrics）

        Returns:
            KnowledgeSubgraph 对象（失败时返回空子图）
        """
        try:
            central_nodes = [dict(record["source"])]           # 核心实体（source 节点）
            connected_nodes = [dict(node) for node in record["nodes"]]  # 连通节点列表
            # 保留关系类型（dict(rel) 只含属性，不含 type），供前端可视化展示走了哪些关系
            relationships = [
                {"type": rel.type, "properties": dict(rel)}
                for rel in record["rels"]
            ]      # 关系列表

            return KnowledgeSubgraph(
                central_nodes=central_nodes,
                connected_nodes=connected_nodes,
                relationships=relationships,
                graph_metrics=record["metrics"],   # 含 node_count / relationship_count / density
                reasoning_chains=[]                # 推理链在后续 graph_structure_reasoning 中填充
            )
        except Exception as e:
            logger.error(f"构建知识子图失败: {e}")
            return KnowledgeSubgraph(
                central_nodes=[],
                connected_nodes=[],
                relationships=[],
                graph_metrics={},
                reasoning_chains=[]
            )

    def _paths_to_documents(self, paths: List[GraphPath], query: str) -> List[Document]:
        """将图路径列表转换为 Document 对象列表。

        每条路径转换为自然语言描述（如"宫保鸡丁 --REQUIRES--> 鸡胸肉"），
        便于 LLM 在生成答案时理解。

        Args:
            paths: GraphPath 列表
            query: 原始查询（当前未使用，预留扩展）

        Returns:
            Document 列表（metadata 含 search_type="graph_path"）
        """
        documents = []

        for i, path in enumerate(paths):
            # 构建路径的自然语言描述（"节点A --关系--> 节点B --关系--> 节点C"）
            path_desc = self._build_path_description(path)

            # 保留结构化路径（供前端可视化推理路径：节点链 + 关系 + 跳数）
            # 只取 name/labels 与 type，剔除大体积 properties，控制 payload 大小。
            structured_nodes = [
                {"name": n.get("name", ""), "labels": list(n.get("labels", []))}
                for n in path.nodes
            ]
            structured_rels = [
                {"type": r.get("type", "")} for r in path.relationships
            ]

            doc = Document(
                page_content=path_desc,
                metadata={
                    "search_type": "graph_path",          # 标记来源为图路径
                    "path_length": path.path_length,       # 路径长度（跳数）
                    "relevance_score": path.relevance_score,  # 路径相关性得分
                    "path_type": path.path_type,           # 路径类型标记
                    "node_count": len(path.nodes),         # 路径上的节点数
                    "relationship_count": len(path.relationships),  # 路径上的关系数
                    "path_nodes": structured_nodes,        # 结构化节点链（可视化用）
                    "path_relationships": structured_rels,  # 结构化关系链（可视化用）
                    "recipe_name": path.nodes[0].get("name", "图结构结果") if path.nodes else "图结构结果"  # 首节点名作为 recipe_name
                }
            )
            documents.append(doc)

        return documents

    def _subgraph_to_documents(self, subgraph: KnowledgeSubgraph,
                              reasoning_chains: List[str], query: str) -> List[Document]:
        """将知识子图转换为 Document 对象。

        将整个子图（中心节点 + 连通节点 + 关系 + 推理链）转换为单个 Document，
        page_content 为子图的自然语言描述，metadata 含图谱指标和推理链。

        Args:
            subgraph: 知识子图对象
            reasoning_chains: 图结构推理生成的推理链列表
            query: 原始查询（当前未使用，预留扩展）

        Returns:
            Document 列表（单元素，metadata 含 search_type="knowledge_subgraph"）
        """
        documents = []

        # 构建子图整体的自然语言描述
        subgraph_desc = self._build_subgraph_description(subgraph)

        # 保留结构化子图（供前端可视化：连通节点名 + 关系类型 + 密度 + 推理链）
        connected_names = [
            n.get("name", "") for n in subgraph.connected_nodes if n.get("name")
        ]
        subgraph_rels = [
            {"type": r.get("type", r.get("relation_type", ""))}
            for r in subgraph.relationships
        ]

        doc = Document(
            page_content=subgraph_desc,
            metadata={
                "search_type": "knowledge_subgraph",              # 标记来源为知识子图
                "node_count": len(subgraph.connected_nodes),      # 连通节点数
                "relationship_count": len(subgraph.relationships),  # 关系数
                "graph_density": subgraph.graph_metrics.get("density", 0.0),  # 图谱密度
                "reasoning_chains": reasoning_chains,             # 推理链列表
                "path_nodes": [{"name": n} for n in connected_names],  # 复用 path_nodes 字段存连通节点（可视化统一处理）
                "path_relationships": subgraph_rels,              # 复用字段存子图关系
                "recipe_name": subgraph.central_nodes[0].get("name", "知识子图") if subgraph.central_nodes else "知识子图"  # 核心实体名
            }
        )
        documents.append(doc)

        return documents

    def _build_path_description(self, path: GraphPath) -> str:
        """构建路径的自然语言描述。

        将路径上的节点和关系拼接为可读的字符串：
            "节点A --关系类型--> 节点B --关系类型--> 节点C"

        Args:
            path: GraphPath 对象

        Returns:
            路径的自然语言描述字符串
        """
        if not path.nodes:
            return "空路径"

        desc_parts = []
        for i, node in enumerate(path.nodes):
            desc_parts.append(node.get("name", f"节点{i}"))
            if i < len(path.relationships):
                rel_type = path.relationships[i].get("type", "相关")
                desc_parts.append(f" --{rel_type}--> ")

        return "".join(desc_parts)

    def _build_subgraph_description(self, subgraph: KnowledgeSubgraph) -> str:
        """构建子图的自然语言描述。

        生成类似"关于 川菜 的知识网络，包含 N 个相关概念和 M 个关系"的描述。

        Args:
            subgraph: KnowledgeSubgraph 对象

        Returns:
            子图的自然语言描述字符串
        """
        central_names = [node.get("name", "未知") for node in subgraph.central_nodes]
        node_count = len(subgraph.connected_nodes)
        rel_count = len(subgraph.relationships)

        return f"关于 {', '.join(central_names)} 的知识网络，包含 {node_count} 个相关概念和 {rel_count} 个关系。"

    def _rank_by_graph_relevance(self, documents: List[Document], query: str) -> List[Document]:
        """基于图结构相关性排序（按 relevance_score 降序）。

        Args:
            documents: 待排序的 Document 列表
            query: 原始查询（当前未使用，预留扩展）

        Returns:
            按 relevance_score 降序排列的 Document 列表
        """
        return sorted(documents,
                     key=lambda x: x.metadata.get("relevance_score", 0.0),
                     reverse=True)

    def _analyze_query_complexity(self, query: str) -> float:
        """分析查询复杂度（基于关键词匹配的简单启发式）。

        通过检测查询中是否包含复杂度指示词（如"什么""如何""为什么""关系"等），
        计算复杂度得分（0-1）。

        Args:
            query: 用户的自然语言查询

        Returns:
            复杂度得分（0-1，值越高表示查询越复杂）
        """
        complexity_indicators = ["什么", "如何", "为什么", "哪些", "关系", "影响", "原因"]
        score = sum(1 for indicator in complexity_indicators if indicator in query)
        return min(score / len(complexity_indicators), 1.0)

    def _identify_reasoning_patterns(self, subgraph: KnowledgeSubgraph) -> List[str]:
        """识别推理模式（当前为固定模式：因果 / 组成 / 相似关系）。

        Args:
            subgraph: 知识子图（当前未使用，预留基于子图拓扑的动态识别）

        Returns:
            推理模式列表
        """
        return ["因果关系", "组成关系", "相似关系"]

    def _build_reasoning_chain(self, pattern: str, subgraph: KnowledgeSubgraph) -> Optional[str]:
        """构建推理链（当前为占位实现）。

        Args:
            pattern: 推理模式（因果 / 组成 / 相似）
            subgraph: 知识子图（当前未使用，预留基于子图内容的推理链构建）

        Returns:
            推理链字符串（占位实现）
        """
        return f"基于{pattern}的推理链"

    def _validate_reasoning_chains(self, chains: List[str], query: str) -> List[str]:
        """验证推理链的可信度（当前为占位实现，返回前 3 条）。

        Args:
            chains: 待验证的推理链列表
            query: 原始查询（当前未使用，预留基于查询的验证逻辑）

        Returns:
            验证后的推理链列表（最多 3 条）
        """
        return chains[:3]

    def _find_entity_relations(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找实体间关系（当前为占位实现，返回空列表）。

        Args:
            graph_query: 图查询计划
            session: Neo4j 会话

        Returns:
            GraphPath 列表（占位实现，返回空）
        """
        return []

    def _find_shortest_paths(self, graph_query: GraphQuery, session) -> List[GraphPath]:
        """查找最短路径（当前为占位实现，返回空列表）。

        Args:
            graph_query: 图查询计划
            session: Neo4j 会话

        Returns:
            GraphPath 列表（占位实现，返回空）
        """
        return []

    def _fallback_subgraph_extraction(self, graph_query: GraphQuery) -> KnowledgeSubgraph:
        """降级子图提取（返回空子图，避免影响主流程）。

        Args:
            graph_query: 图查询计划（当前未使用）

        Returns:
            空的 KnowledgeSubgraph 对象
        """
        return KnowledgeSubgraph(
            central_nodes=[],
            connected_nodes=[],
            relationships=[],
            graph_metrics={},
            reasoning_chains=[]
        )

    def close(self):
        """关闭 Neo4j 资源连接。"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("图RAG检索系统已关闭")