"""
智能查询路由器 - 根据查询特点自动选择最适合的检索策略

核心职责：
    分析用户查询的特征（复杂度、关系密集度、推理需求等），
    自动路由到最适合的检索引擎：
        - 传统混合检索（hybrid_traditional）：适合简单的信息查找
        - 图 RAG 检索（graph_rag）：适合复杂的关系推理和知识发现
        - 组合策略（combined）：需要两种策略结合的复杂查询

路由决策流程：
    query
      ↓ (analyze_query - LLM 智能分析)
    QueryAnalysis (复杂度 / 关系密集度 / 推理需求 / 推荐策略)
      ↓ (route_query - 根据策略分派)
    ┌──────────────────┬──────────────┬──────────────┐
    │ hybrid_traditional │ graph_rag     │ combined     │
    ↓                   ↓              ↓
    HybridRetrieval     GraphRAG       _combined_search
    .hybrid_search      .graph_rag_search (Round-robin 交替)

外部依赖：
    - HybridRetrievalModule: 传统混合检索引擎
    - GraphRAGRetrieval: 图 RAG 检索引擎
    - LLM: 用于查询特征分析（降级时切换为基于规则的分析）
"""

import json
import logging
import hashlib
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """搜索策略枚举：路由器可选的检索策略。

    Attributes:
        HYBRID_TRADITIONAL: 传统混合检索（BM25 + 向量 + 图键值，RRF 融合）
        GRAPH_RAG: 图 RAG 检索（多跳遍历 + 子图提取 + 图结构推理）
        COMBINED: 组合策略（两种策略 Round-robin 交替融合）
    """
    HYBRID_TRADITIONAL = "hybrid_traditional"  # 传统混合检索
    GRAPH_RAG = "graph_rag"                    # 图 RAG 检索
    COMBINED = "combined"                      # 组合策略


@dataclass
class QueryAnalysis:
    """查询分析结果：封装 LLM 对查询特征的深度分析。

    Attributes:
        query_complexity: 查询复杂度 (0-1)，0=简单信息查找，1=高复杂度推理
        relationship_intensity: 关系密集度 (0-1)，0=单一实体，1=复杂关系网络
        reasoning_required: 是否需要多跳推理 / 因果分析 / 对比分析
        entity_count: 查询中包含的明确实体数量
        recommended_strategy: 推荐的检索策略（HYBRID_TRADITIONAL / GRAPH_RAG / COMBINED）
        confidence: 推荐置信度 (0-1)
        reasoning: 推荐理由（自然语言说明）
    """
    query_complexity: float           # 查询复杂度 (0-1)
    relationship_intensity: float     # 关系密集度 (0-1)
    reasoning_required: bool          # 是否需要推理
    entity_count: int                 # 实体数量
    recommended_strategy: SearchStrategy
    confidence: float                 # 推荐置信度
    reasoning: str                    # 推荐理由


class IntelligentQueryRouter:
    """智能查询路由器

    核心能力：
        1. 查询复杂度分析：识别简单查找 vs 复杂推理
        2. 关系密集度评估：判断是否需要图结构优势
        3. 策略自动选择：路由到最适合的检索引擎
        4. 结果质量监控：基于反馈优化路由决策（通过 route_stats 统计）

    Public API:
        - route_query(query, top_k)             -> 主路由接口，返回 (documents, analysis)
        - explain_routing_decision(query)       -> 解释路由决策（供调试）
        - get_route_statistics()                -> 获取路由统计（各策略使用占比）

    路由决策依据（LLM 分析的四个维度）：
        - query_complexity: 0-0.3 简单 / 0.4-0.7 中等 / 0.8-1.0 复杂
        - relationship_intensity: 0-0.3 单一实体 / 0.4-0.7 实体关系 / 0.8-1.0 复杂网络
        - reasoning_required: 是否需要多跳/因果/对比推理
        - entity_count: 查询中的明确实体数

    Note:
        LLM 分析失败时降级为基于规则的分析（_rule_based_analysis），
        通过关键词匹配估算复杂度和关系密集度。
    """

    def __init__(self,
                 traditional_retrieval,  # 传统混合检索模块
                 graph_rag_retrieval,    # 图 RAG 检索模块
                 llm_client,
                 config):
        """初始化智能查询路由器。

        Args:
            traditional_retrieval: HybridRetrievalModule 实例（传统混合检索）
            graph_rag_retrieval: GraphRAGRetrieval 实例（图 RAG 检索）
            llm_client: LLM 客户端（用于查询特征分析）
            config: GraphRAGConfig 配置对象
        """
        self.traditional_retrieval = traditional_retrieval
        self.graph_rag_retrieval = graph_rag_retrieval
        self.llm_client = llm_client
        self.config = config

        # 路由统计：记录各策略的使用次数，供后续分析和优化
        self.route_stats = {
            "traditional_count": 0,  # 传统混合检索使用次数
            "graph_rag_count": 0,    # 图 RAG 检索使用次数
            "combined_count": 0,     # 组合策略使用次数
            "total_queries": 0       # 总查询次数
        }

    def analyze_query(self, query: str) -> QueryAnalysis:
        """深度分析查询特征，决定最佳检索策略。

        使用 LLM 从四个维度分析查询：
            1. 查询复杂度（0-1）：简单信息查找 vs 高复杂度推理
            2. 关系密集度（0-1）：单一实体 vs 复杂关系网络
            3. 推理需求：是否需要多跳/因果/对比推理
            4. 实体识别：查询中的明确实体数量和类型

        基于分析结果推荐检索策略：
            - hybrid_traditional: 适合简单直接的信息查找
            - graph_rag: 适合复杂关系推理和知识发现
            - combined: 需要两种策略结合

        Args:
            query: 用户的自然语言查询

        Returns:
            QueryAnalysis 对象（LLM 失败时降级为基于规则的分析）
        """
        logger.info(f"分析查询特征: {query}")

        # 使用 LLM 进行智能分析（提示词中包含分级示例和 JSON 输出格式要求）
        analysis_prompt = f"""
        作为RAG系统的查询分析专家，请深度分析以下查询的特征：

        查询：{query}

        请从以下维度分析：

        1. 查询复杂度 (0-1)：
           - 0.0-0.3: 简单信息查找（如：红烧肉怎么做？）
           - 0.4-0.7: 中等复杂度（如：川菜有哪些特色菜？）
           - 0.8-1.0: 高复杂度推理（如：为什么川菜用花椒而不是胡椒？）

        2. 关系密集度 (0-1)：
           - 0.0-0.3: 单一实体信息（如：西红柿的营养价值）
           - 0.4-0.7: 实体间关系（如：鸡肉配什么蔬菜？）
           - 0.8-1.0: 复杂关系网络（如：川菜的形成与地理、历史的关系）

        3. 推理需求：
           - 是否需要多跳推理？
           - 是否需要因果分析？
           - 是否需要对比分析？

        4. 实体识别：
           - 查询中包含多少个明确实体？
           - 实体类型是什么？

        基于分析推荐检索策略：
        - hybrid_traditional: 适合简单直接的信息查找
        - graph_rag: 适合复杂关系推理和知识发现
        - combined: 需要两种策略结合

        返回JSON格式：
        {{
            "query_complexity": 0.6,
            "relationship_intensity": 0.8,
            "reasoning_required": true,
            "entity_count": 3,
            "recommended_strategy": "graph_rag",
            "confidence": 0.85,
            "reasoning": "该查询涉及多个实体间的复杂关系，需要图结构推理"
        }}
        """

        try:
            response = self.llm_client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,  # 低温度确保分析结果稳定
                max_tokens=800
            )

            result = json.loads(response.choices[0].message.content.strip())

            # 将 LLM 返回的 JSON 转换为 QueryAnalysis 对象
            analysis = QueryAnalysis(
                query_complexity=result.get("query_complexity", 0.5),
                relationship_intensity=result.get("relationship_intensity", 0.5),
                reasoning_required=result.get("reasoning_required", False),
                entity_count=result.get("entity_count", 1),
                recommended_strategy=SearchStrategy(result.get("recommended_strategy", "hybrid_traditional")),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "默认分析")
            )

            logger.info(f"查询分析完成: {analysis.recommended_strategy.value} (置信度: {analysis.confidence:.2f})")
            return analysis

        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            # 降级方案：基于规则的简单分析（不依赖 LLM）
            return self._rule_based_analysis(query)

    def _rule_based_analysis(self, query: str) -> QueryAnalysis:
        """基于规则的降级分析（LLM 不可用时的后备方案）。

        通过关键词匹配估算复杂度和关系密集度：
            - 复杂度关键词：为什么 / 如何 / 关系 / 影响 / 原因 / 比较 / 区别
            - 关系关键词：配 / 搭配 / 组合 / 相关 / 联系 / 连接

        Args:
            query: 用户的自然语言查询

        Returns:
            QueryAnalysis 对象（置信度固定为 0.6，标记为"基于规则的简单分析"）
        """
        # 复杂度关键词：出现这些词通常表示需要推理
        complexity_keywords = ["为什么", "如何", "关系", "影响", "原因", "比较", "区别"]
        # 关系关键词：出现这些词通常表示涉及实体间关系
        relation_keywords = ["配", "搭配", "组合", "相关", "联系", "连接"]

        # 计算复杂度和关系密集度（基于关键词命中比例）
        complexity = sum(1 for kw in complexity_keywords if kw in query) / len(complexity_keywords)
        relation_intensity = sum(1 for kw in relation_keywords if kw in query) / len(relation_keywords)

        # 策略选择：复杂度或关系密集度超过阈值时使用图 RAG
        if complexity > 0.3 or relation_intensity > 0.3:
            strategy = SearchStrategy.GRAPH_RAG
        else:
            strategy = SearchStrategy.HYBRID_TRADITIONAL

        return QueryAnalysis(
            query_complexity=complexity,
            relationship_intensity=relation_intensity,
            reasoning_required=complexity > 0.3,
            entity_count=len(query.split()),
            recommended_strategy=strategy,
            confidence=0.6,  # 规则分析的置信度较低
            reasoning="基于规则的简单分析"
        )

    def route_query(self, query: str, top_k: int = 5) -> Tuple[List[Document], QueryAnalysis]:
        """智能路由查询到最适合的检索引擎。

        执行流程：
            1. 分析查询特征（analyze_query）-> QueryAnalysis
            2. 更新路由统计（_update_route_stats）
            3. 根据推荐策略分派到对应检索引擎
            4. 结果后处理（_post_process_results：添加路由元信息）

        Args:
            query: 用户的自然语言查询
            top_k: 返回结果数量上限

        Returns:
            (documents, analysis) 元组：
                - documents: 检索结果 Document 列表
                - analysis: 查询分析结果（含推荐策略和置信度）

        Note:
            任何检索引擎失败时降级到传统混合检索，确保系统可用性。
        """
        logger.info(f"开始智能路由: {query}")

        # 1. 分析查询特征（LLM 智能分析或规则降级）
        analysis = self.analyze_query(query)

        # 2. 更新路由统计（记录各策略使用次数）
        self._update_route_stats(analysis.recommended_strategy)

        # 3. 根据策略执行检索
        documents = []

        try:
            if analysis.recommended_strategy == SearchStrategy.HYBRID_TRADITIONAL:
                # 传统混合检索：BM25 + 向量 + 图键值，RRF 融合
                logger.info("使用传统混合检索")
                documents = self.traditional_retrieval.hybrid_search(query, top_k)

            elif analysis.recommended_strategy == SearchStrategy.GRAPH_RAG:
                # 图 RAG 检索：多跳遍历 + 子图提取 + 图结构推理
                logger.info("🕸️ 使用图RAG检索")
                documents = self.graph_rag_retrieval.graph_rag_search(query, top_k)

            elif analysis.recommended_strategy == SearchStrategy.COMBINED:
                # 组合策略：两种策略 Round-robin 交替融合
                logger.info("🔄 使用组合检索策略")
                documents = self._combined_search(query, top_k)

            # 4. 结果后处理：添加路由元信息（route_strategy / query_complexity / route_confidence）
            documents = self._post_process_results(documents, analysis)

            logger.info(f"路由完成，返回 {len(documents)} 个结果")
            return documents, analysis

        except Exception as e:
            logger.error(f"查询路由失败: {e}")
            # 降级到传统检索（确保系统可用性）
            documents = self.traditional_retrieval.hybrid_search(query, top_k)
            return documents, analysis

    def _combined_search(self, query: str, top_k: int) -> List[Document]:
        """组合搜索策略：结合传统检索和图 RAG 的优势。

        采用 Round-robin（轮询）交替融合策略：
            1. 将 top_k 平分给两种策略（traditional_k + graph_k = top_k）
            2. 交替添加图 RAG 结果和传统检索结果（图 RAG 优先，通常质量更高）
            3. 基于内容哈希去重（page_content[:100] 的 MD5）

        Args:
            query: 用户的自然语言查询
            top_k: 最终返回的文档数量

        Returns:
            融合后的 Document 列表（metadata 含 search_source 标记来源）
        """
        # 分配结果数量：传统检索和图 RAG 各占一半
        traditional_k = max(1, top_k // 2)
        graph_k = top_k - traditional_k

        # 执行两种检索
        traditional_docs = self.traditional_retrieval.hybrid_search(query, traditional_k)
        graph_docs = self.graph_rag_retrieval.graph_rag_search(query, graph_k)

        # 合并和去重（基于内容哈希）
        combined_docs = []
        seen_contents = set()  # 已添加内容的哈希集合

        # 交替添加结果（Round-robin）：图 RAG 优先（通常质量更高）
        max_len = max(len(traditional_docs), len(graph_docs))
        for i in range(max_len):
            # 先添加图 RAG 结果（通常质量更高，优先保留）
            if i < len(graph_docs):
                doc = graph_docs[i]
                # 基于内容前 100 字符的 MD5 去重
                content_hash = hashlib.md5(doc.page_content[:100].encode('utf-8')).hexdigest()
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "graph_rag"  # 标记来源
                    combined_docs.append(doc)

            # 再添加传统检索结果
            if i < len(traditional_docs):
                doc = traditional_docs[i]
                content_hash = hashlib.md5(doc.page_content[:100].encode('utf-8')).hexdigest()
                if content_hash not in seen_contents:
                    seen_contents.add(content_hash)
                    doc.metadata["search_source"] = "traditional"  # 标记来源
                    combined_docs.append(doc)

        return combined_docs[:top_k]

    def _post_process_results(self, documents: List[Document], analysis: QueryAnalysis) -> List[Document]:
        """结果后处理：将路由分析信息写入文档元数据。

        为每个文档添加路由元信息，便于下游分析和调试：
            - route_strategy: 使用的检索策略
            - query_complexity: 查询复杂度
            - route_confidence: 路由置信度

        Args:
            documents: 待处理的 Document 列表
            analysis: 查询分析结果

        Returns:
            处理后的 Document 列表（元数据中包含路由信息）
        """
        for doc in documents:
            # 添加路由信息到元数据（供下游展示和分析使用）
            doc.metadata.update({
                "route_strategy": analysis.recommended_strategy.value,
                "query_complexity": analysis.query_complexity,
                "route_confidence": analysis.confidence
            })

        return documents

    def _update_route_stats(self, strategy: SearchStrategy):
        """更新路由统计（记录各策略的使用次数）。

        Args:
            strategy: 本次查询使用的检索策略
        """
        self.route_stats["total_queries"] += 1

        if strategy == SearchStrategy.HYBRID_TRADITIONAL:
            self.route_stats["traditional_count"] += 1
        elif strategy == SearchStrategy.GRAPH_RAG:
            self.route_stats["graph_rag_count"] += 1
        elif strategy == SearchStrategy.COMBINED:
            self.route_stats["combined_count"] += 1

    def get_route_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息（含各策略使用次数和占比）。

        Returns:
            包含以下信息的字典：
                - total_queries / traditional_count / graph_rag_count / combined_count: 各策略使用次数
                - traditional_ratio / graph_rag_ratio / combined_ratio: 各策略使用占比

            若 total_queries=0，则直接返回原始 route_stats（无占比计算）。
        """
        total = self.route_stats["total_queries"]
        if total == 0:
            return self.route_stats

        # 计算各策略使用占比
        return {
            **self.route_stats,
            "traditional_ratio": self.route_stats["traditional_count"] / total,
            "graph_rag_ratio": self.route_stats["graph_rag_count"] / total,
            "combined_ratio": self.route_stats["combined_count"] / total
        }

    def explain_routing_decision(self, query: str) -> str:
        """解释路由决策过程（供调试和可解释性展示）。

        生成一份路由分析报告，包含：
            - 查询内容
            - 特征分析（复杂度、关系密集度、推理需求、实体数量）
            - 推荐策略和置信度
            - 决策理由

        Args:
            query: 用户的自然语言查询

        Returns:
            路由决策报告字符串（含复杂度和关系密集度的分级描述）
        """
        analysis = self.analyze_query(query)

        # 生成路由决策报告（含分级描述：简单/中等/复杂）
        explanation = f"""
        查询路由分析报告

        查询：{query}

        特征分析：
        - 复杂度：{analysis.query_complexity:.2f} ({'简单' if analysis.query_complexity < 0.4 else '中等' if analysis.query_complexity < 0.8 else '复杂'})
        - 关系密集度：{analysis.relationship_intensity:.2f} ({'单一实体' if analysis.relationship_intensity < 0.4 else '实体关系' if analysis.relationship_intensity < 0.8 else '复杂关系网络'})
        - 推理需求：{'是' if analysis.reasoning_required else '否'}
        - 实体数量：{analysis.entity_count}

        推荐策略：{analysis.recommended_strategy.value}
        置信度：{analysis.confidence:.2f}

        决策理由：{analysis.reasoning}
        """

        return explanation
