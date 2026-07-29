"""
基于图 RAG 的智能烹饪助手 - 主程序（系统编排层）

整合传统检索和图 RAG 检索，实现真正的图数据优势。
本文件是整个 RAG 系统的启动入口和编排层，负责串联各核心模块。

系统架构（数据流向）：
    ┌─────────────────────────────────────────────────────────────────┐
    │                     AdvancedGraphRAGSystem                      │
    │                                                                 │
    │  Neo4j ──> GraphDataPreparationModule ──> Document 列表 ──> 分块 │
    │                                              ↓                  │
    │                              MilvusIndexConstructionModule       │
    │                                     ↓ (向量索引)                 │
    │  用户查询 ──> IntelligentQueryRouter (LLM 路由决策)              │
    │                     ↓                                           │
    │      ┌──────────────┴───────────────┐                          │
    │      ↓                              ↓                           │
    │  HybridRetrievalModule       GraphRAGRetrieval                  │
    │  (BM25+向量+图键值,RRF)      (多跳遍历+子图推理)                 │
    │      └──────────────┬───────────────┘                          │
    │                     ↓                                           │
    │          GenerationIntegrationModule (LLM 生成答案)             │
    └─────────────────────────────────────────────────────────────────┘

启动流程：
    main() -> AdvancedGraphRAGSystem()
           -> initialize_system()     # 初始化 6 大核心模块
           -> build_knowledge_base()  # 构建或加载知识库
           -> run_interactive()       # 进入交互式问答循环
"""

import os
import sys
import time
import logging
from typing import List, Optional

# 设置日志格式（时间 - 模块名 - 级别 - 消息）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加当前目录到 Python 路径，确保 rag_modules 包能被正确导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, GraphRAGConfig
from rag_modules import (
    GraphDataPreparationModule,
    MilvusIndexConstructionModule,
    GenerationIntegrationModule,
    RAGASEvaluationModule
)
from rag_modules.hybrid_retrieval import HybridRetrievalModule
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval
from rag_modules.intelligent_query_router import IntelligentQueryRouter, QueryAnalysis
from rag_modules.markdown_parser import parse_markdown_recipe
from langchain_core.documents import Document

# 加载环境变量（如 DEEPSEEK_API_KEY 等）
load_dotenv()

class AdvancedGraphRAGSystem:
    """图 RAG 系统 - 整合传统检索与图 RAG 检索的统一编排层

    核心特性：
        1. 智能路由：自动选择最适合的检索策略（传统混合 / 图 RAG / 组合）
        2. 双引擎检索：传统混合检索（BM25+向量+图键值）+ 图 RAG 检索（多跳遍历）
        3. 图结构推理：多跳遍历、子图提取、关系推理
        4. 查询复杂度分析：深度理解用户意图（LLM 智能分析）
        5. 自适应学习：基于反馈优化系统性能（路由统计）

    模块组成（initialize_system 中初始化）：
        - data_module:          GraphDataPreparationModule  (Neo4j -> Document)
        - index_module:         MilvusIndexConstructionModule (Document -> 向量索引)
        - generation_module:    GenerationIntegrationModule  (LLM 答案生成)
        - traditional_retrieval: HybridRetrievalModule       (传统混合检索)
        - graph_rag_retrieval:  GraphRAGRetrieval            (图 RAG 检索)
        - query_router:         IntelligentQueryRouter       (智能路由器)
    """

    def __init__(self, config: Optional[GraphRAGConfig] = None):
        """初始化图 RAG 系统（仅设置配置，不建立连接）。

        Args:
            config: 系统配置对象（None 时使用 DEFAULT_CONFIG）
        """
        self.config = config or DEFAULT_CONFIG

        # 核心模块（在 initialize_system 中实例化）
        self.data_module = None          # 数据准备模块（Neo4j -> Document）
        self.index_module = None         # 向量索引模块（Milvus）
        self.generation_module = None    # 生成模块（LLM）

        # 检索引擎（在 initialize_system 中实例化）
        self.traditional_retrieval = None  # 传统混合检索
        self.graph_rag_retrieval = None    # 图 RAG 检索
        self.query_router = None           # 智能路由器

        # 评估引擎（懒加载 ragas/LLM/embeddings，启动零成本）
        self.evaluation_module = None     # RAGAS 评估模块

        # 系统状态：标记知识库是否已就绪（可接受查询）
        self.system_ready = False

    def initialize_system(self):
        """初始化高级图 RAG 系统：实例化 6 大核心模块。

        初始化顺序（存在依赖关系，不可随意调整）：
            1. 数据准备模块（独立）
            2. 向量索引模块（独立）
            3. 生成模块（独立，提供 LLM 客户端供下游使用）
            4. 传统混合检索（依赖 1/2/3：需要 data_module、index_module、llm_client）
            5. 图 RAG 检索（依赖 3：需要 llm_client）
            6. 智能查询路由器（依赖 4/5：需要两个检索引擎和 llm_client）

        Raises:
            Exception: 任何模块初始化失败时抛出异常
        """
        logger.info("启动高级图RAG系统...")

        try:
            # 1. 数据准备模块：连接 Neo4j，负责图数据加载和文档构建
            print("初始化数据准备模块...")
            self.data_module = GraphDataPreparationModule(
                uri=self.config.neo4j_uri,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password,
                database=self.config.neo4j_database
            )

            # 2. 向量索引模块：连接 Milvus，负责向量化和索引构建
            print("初始化Milvus向量索引...")
            self.index_module = MilvusIndexConstructionModule(
                host=self.config.milvus_host,
                port=self.config.milvus_port,
                collection_name=self.config.milvus_collection_name,
                dimension=self.config.milvus_dimension,
                model_name=self.config.embedding_model
            )

            # 3. 生成模块：初始化 LLM 客户端（DeepSeek API），供下游检索器复用
            print("初始化生成模块...")
            self.generation_module = GenerationIntegrationModule(
                model_name=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )

            # 4. 传统混合检索模块：BM25 + 向量 + 图键值，RRF 融合
            print("初始化传统混合检索...")
            self.traditional_retrieval = HybridRetrievalModule(
                config=self.config,
                milvus_module=self.index_module,       # 依赖向量索引模块
                data_module=self.data_module,           # 依赖数据准备模块
                llm_client=self.generation_module.client  # 复用 LLM 客户端
            )

            # 5. 图 RAG 检索模块：多跳遍历 + 子图提取 + 图结构推理
            print("初始化图RAG检索引擎...")
            self.graph_rag_retrieval = GraphRAGRetrieval(
                config=self.config,
                llm_client=self.generation_module.client  # 复用 LLM 客户端
            )

            # 6. 智能查询路由器：根据查询特征自动选择检索策略
            print("初始化智能查询路由器...")
            self.query_router = IntelligentQueryRouter(
                traditional_retrieval=self.traditional_retrieval,  # 传统检索引擎
                graph_rag_retrieval=self.graph_rag_retrieval,      # 图 RAG 引擎
                llm_client=self.generation_module.client,          # 复用 LLM 客户端
                config=self.config
            )

            # 7. RAGAS 评估引擎：仅存配置，ragas/LLM/embeddings 懒加载（首次评估才初始化）
            print("初始化RAGAS评估引擎...")
            self.evaluation_module = RAGASEvaluationModule(self.config)

            print("✅ 高级图RAG系统初始化完成！")

        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            raise

    def build_knowledge_base(self):
        """构建知识库（若已存在则加载，否则从 Neo4j 重新构建）。

        两种路径：
            路径 A（已存在集合）：加载 Milvus 集合 -> 重新加载图数据 ->
                                 构建菜谱文档 -> 分块 -> 初始化检索器
            路径 B（新建）：从 Neo4j 加载图数据 -> 构建菜谱文档 -> 分块 ->
                           构建向量索引 -> 初始化检索器 -> 显示统计

        Note:
            即使从已存在的知识库加载，也需要重新加载图数据（load_graph_data），
            因为图索引（GraphIndexingModule）依赖内存中的图节点数据。
        """
        print("\n检查知识库状态...")

        try:
            # 检查 Milvus 集合是否已存在（避免重复构建）
            if self.index_module.has_collection():
                print("✅ 发现已存在的知识库，尝试加载...")
                if self.index_module.load_collection():
                    print("知识库加载成功！")

                    # 重要：即使从已存在的知识库加载，也需要加载图数据以支持图索引
                    # （图索引在内存中构建，依赖 data_module.recipes/ingredients/cooking_steps）
                    print("加载图数据以支持图检索...")
                    self.data_module.load_graph_data()
                    print("构建菜谱文档...")
                    self.data_module.build_recipe_documents()
                    print("进行文档分块...")
                    chunks = self.data_module.chunk_documents(
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap
                    )

                    # 初始化检索器（BM25 索引 + 图索引）
                    self._initialize_retrievers(chunks)
                    return
                else:
                    print("❌ 知识库加载失败，开始重建...")

            print("未找到已存在的集合，开始构建新的知识库...")

            # 从 Neo4j 加载图数据（Recipe / Ingredient / CookingStep 节点）
            print("从Neo4j加载图数据...")
            self.data_module.load_graph_data()

            # 构建菜谱文档（将图节点组装为自然语言描述的 Document）
            print("构建菜谱文档...")
            self.data_module.build_recipe_documents()

            # 进行文档分块（按章节或固定长度）
            print("进行文档分块...")
            chunks = self.data_module.chunk_documents(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )

            # 构建 Milvus 向量索引（向量化 + 插入 + 建索引）
            print("构建Milvus向量索引...")
            if not self.index_module.build_vector_index(chunks):
                raise Exception("构建向量索引失败")

            # 初始化检索器（BM25 索引 + 图索引）
            self._initialize_retrievers(chunks)

            # 显示知识库统计信息（节点数、向量数、分类等）
            self._show_knowledge_base_stats()

            print("✅ 知识库构建完成！")

        except Exception as e:
            logger.error(f"知识库构建失败: {e}")
            raise

    def _initialize_retrievers(self, chunks: List = None):
        """初始化检索器：构建 BM25 索引和图索引。

        Args:
            chunks: 文档分块列表（None 时从 data_module.chunks 获取）
        """
        print("初始化检索引擎...")

        # 如果没有 chunks，从数据模块获取（已分块的结果）
        if chunks is None:
            chunks = self.data_module.chunks or []

        # 初始化传统检索器（BM25 索引 + 图键值索引 + 父文档映射）
        self.traditional_retrieval.initialize(chunks)

        # 初始化图 RAG 检索器（连接 Neo4j + 预热图索引）
        self.graph_rag_retrieval.initialize()

        self.system_ready = True  # 标记系统就绪，可接受查询
        print("✅ 检索引擎初始化完成！")

    def _show_knowledge_base_stats(self):
        """显示知识库统计信息（节点数、向量数、路由统计、主要分类）。"""
        print(f"\n知识库统计:")

        # 数据统计（菜谱数、食材数、步骤数、文档数、分块数）
        stats = self.data_module.get_statistics()
        print(f"   菜谱数量: {stats.get('total_recipes', 0)}")
        print(f"   食材数量: {stats.get('total_ingredients', 0)}")
        print(f"   烹饪步骤: {stats.get('total_cooking_steps', 0)}")
        print(f"   文档数量: {stats.get('total_documents', 0)}")
        print(f"   文本块数: {stats.get('total_chunks', 0)}")

        # Milvus 统计（向量索引记录数）
        milvus_stats = self.index_module.get_collection_stats()
        print(f"   向量索引: {milvus_stats.get('row_count', 0)} 条记录")

        # 图 RAG 统计（路由统计，初始化后通常为 0）
        route_stats = self.query_router.get_route_statistics()
        print(f"   路由统计: 总查询 {route_stats.get('total_queries', 0)} 次")

        # 显示主要分类（最多 10 个）
        if stats.get('categories'):
            categories = list(stats['categories'].keys())[:10]
            print(f"   🏷️ 主要分类: {', '.join(categories)}")

    def ask_question_with_routing(self, question: str, stream: bool = False, explain_routing: bool = False):
        """智能问答：自动选择最佳检索策略并生成回答。

        执行流程：
            1. （可选）显示路由决策解释（explain_routing=True 时）
            2. 智能路由检索（query_router.route_query）-> 相关文档 + 查询分析
            3. 显示路由信息（策略图标、复杂度、关系密集度）
            4. 显示检索结果信息（菜谱名、搜索类型、得分）
            5. 生成回答（流式或标准模式）
            6. 性能统计（耗时）

        Args:
            question: 用户的问题
            stream: 是否使用流式输出（默认 False）
            explain_routing: 是否显示路由决策解释（默认 False）

        Returns:
            (result, analysis) 元组：
                - result: 生成的回答字符串（流式时为"流式输出完成"）
                - analysis: QueryAnalysis 对象（含推荐策略、复杂度等），无结果时为 None

        Raises:
            ValueError: 系统未就绪时（需先构建知识库）
        """
        if not self.system_ready:
            raise ValueError("系统未就绪，请先构建知识库")

        print(f"\n❓ 用户问题: {question}")

        # 显示路由决策解释（可选，供调试和可解释性展示）
        if explain_routing:
            explanation = self.query_router.explain_routing_decision(question)
            print(explanation)

        start_time = time.time()

        try:
            # 1. 智能路由检索：LLM 分析查询特征 -> 选择策略 -> 执行检索
            print("执行智能查询路由...")
            relevant_docs, analysis = self.query_router.route_query(question, self.config.top_k)

            # 2. 显示路由信息（策略图标 + 复杂度 + 关系密集度）
            strategy_icons = {
                "hybrid_traditional": "🔍",  # 传统混合检索
                "graph_rag": "🕸️",          # 图 RAG 检索
                "combined": "🔄"             # 组合策略
            }
            strategy_icon = strategy_icons.get(analysis.recommended_strategy.value, "❓")
            print(f"{strategy_icon} 使用策略: {analysis.recommended_strategy.value}")
            print(f"📊 复杂度: {analysis.query_complexity:.2f}, 关系密集度: {analysis.relationship_intensity:.2f}")

            # 3. 显示检索结果信息（菜谱名、搜索类型、得分）
            if relevant_docs:
                doc_info = []
                for doc in relevant_docs:
                    recipe_name = doc.metadata.get('recipe_name', '未知内容')
                    search_type = doc.metadata.get('search_type', doc.metadata.get('route_strategy', 'unknown'))
                    score = doc.metadata.get('final_score', doc.metadata.get('relevance_score', 0))
                    doc_info.append(f"{recipe_name}({search_type}, {score:.3f})")

                print(f"📋 找到 {len(relevant_docs)} 个相关文档: {', '.join(doc_info[:3])}")
                if len(doc_info) > 3:
                    print(f"    等 {len(relevant_docs)} 个结果...")
            else:
                # 保持返回值签名一致：始终返回 (result, analysis)
                return "抱歉，没有找到相关的烹饪信息。请尝试其他问题。", analysis

            # 4. 生成回答（流式或标准模式）
            print("🎯 智能生成回答...")

            if stream:
                try:
                    # 流式输出：逐字打印，提升用户体验
                    for chunk_text in self.generation_module.generate_adaptive_answer_stream(question, relevant_docs):
                        print(chunk_text, end="", flush=True)
                    print("\n")
                    result = "流式输出完成"
                except Exception as stream_error:
                    logger.error(f"流式输出过程中出现错误: {stream_error}")
                    print(f"\n⚠️ 流式输出中断，切换到标准模式...")
                    # 使用非流式作为后备（确保系统可用性）
                    result = self.generation_module.generate_adaptive_answer(question, relevant_docs)
            else:
                # 标准模式：一次性返回完整答案
                result = self.generation_module.generate_adaptive_answer(question, relevant_docs)

            # 5. 性能统计（总耗时）
            end_time = time.time()
            print(f"\n⏱️ 问答完成，耗时: {end_time - start_time:.2f}秒")

            return result, analysis

        except Exception as e:
            logger.error(f"问答处理失败: {e}")
            return f"抱歉，处理问题时出现错误：{str(e)}", None

    # ------------------------------------------------------------------
    # Web API 支持方法（数据返回型，不 print / 不 input，供 FastAPI 调用）
    # ------------------------------------------------------------------

    def retrieve(self, question: str, top_k: Optional[int] = None):
        """仅执行路由 + 检索，不生成答案（供 Web API 获取来源与分析）。

        Args:
            question: 用户问题
            top_k: 返回结果数量上限（None 时使用 config.top_k）

        Returns:
            (documents, analysis) 元组：
                - documents: 检索结果 Document 列表
                - analysis: QueryAnalysis 对象

        Raises:
            ValueError: 系统未就绪时
        """
        if not self.system_ready:
            raise ValueError("系统未就绪，请先构建知识库")

        return self.query_router.route_query(question, top_k or self.config.top_k)

    def run_dataset_evaluation(self, items: List) -> dict:
        """对测试集跑完整 RAG 管线 + RAGAS 评估（供 Web API 调用）。

        对每条 item（question + ground_truth）：路由检索 -> 生成答案 -> 取 contexts，
        收集后一次性 RAGAS 评估（有 ground_truth，跑完整 4 指标）。

        Args:
            items: [{question, ground_truth, ...}]，ground_truth 可空（空则该条跳过需 gt 的指标）

        Returns:
            {results, aggregates, count, metrics, elapsed, skipped}
        """
        if not self.system_ready:
            raise ValueError("系统未就绪，请先构建知识库")
        if self.evaluation_module is None:
            raise RuntimeError("评估引擎未初始化")

        start = time.time()
        samples = []
        skipped = 0
        for idx, it in enumerate(items, 1):
            q = (it.get("question") or "").strip() if isinstance(it, dict) else ""
            gt = it.get("ground_truth") if isinstance(it, dict) else None
            if not q:
                skipped += 1
                continue
            logger.info(f"评估样本 {idx}/{len(items)}: {q[:50]}...")
            # 跑完整 RAG 管线：路由检索 -> 生成答案（与 /api/query 使用完全相同的管线）
            documents, analysis = self.query_router.route_query(q, self.config.top_k)
            if not documents:
                answer = "（无检索结果）"
                contexts = []
            else:
                answer = self.generation_module.generate_adaptive_answer(q, documents)
                # 收集 contexts 时去重（同一 node_id 的多个 chunk 内容可能重复），
                # 并过滤空串，避免 RAGAS judge 被重复上下文干扰评分。
                seen_content_hashes = set()
                contexts = []
                for d in documents:
                    content = d.page_content.strip()
                    if not content:
                        continue
                    content_hash = hash(content)
                    if content_hash in seen_content_hashes:
                        continue
                    seen_content_hashes.add(content_hash)
                    contexts.append(content)

                # 若生成返回了错误字符串，记录告警（说明 LLM 调用失败，该样本分数会异常低）
                if answer.startswith("抱歉，生成回答时出现错误"):
                    logger.warning(f"样本 {idx} 生成失败: {answer[:100]}")

            samples.append({
                "question": q,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": gt,
            })

        result = self.evaluation_module.evaluate(
            samples,
            # 全部样本都有 ground_truth 才跑完整 4 指标；否则仅忠实度 + 相关性
            with_reference=bool(samples) and all(s.get("ground_truth") for s in samples),
        )
        result["elapsed"] = round(time.time() - start, 2)
        result["skipped"] = skipped
        return result

    def evaluate_message(self, question: str, answer: str) -> dict:
        """对一条存储的问答做在线评估（供 Web API 调用）。

        重新检索取完整 page_content 作为 contexts（存储的 sources 仅有 300 字预览），
        用存储的 answer 评估。无 ground_truth -> 仅忠实度 + 答案相关性。

        Args:
            question: 原问题
            answer: 存储的助手回答

        Returns:
            {results, aggregates, count, metrics}
        """
        if not self.system_ready:
            raise ValueError("系统未就绪，请先构建知识库")
        if self.evaluation_module is None:
            raise RuntimeError("评估引擎未初始化")

        documents, _ = self.query_router.route_query(question, self.config.top_k)
        contexts = [d.page_content.strip() for d in documents if d.page_content.strip()]
        return self.evaluation_module.evaluate_single(question, answer, contexts, ground_truth=None)

    @staticmethod
    def analysis_to_dict(analysis) -> dict:
        """将 QueryAnalysis（含 SearchStrategy 枚举）序列化为可 JSON 化的字典。"""
        return {
            "recommended_strategy": analysis.recommended_strategy.value,
            "query_complexity": analysis.query_complexity,
            "relationship_intensity": analysis.relationship_intensity,
            "reasoning_required": analysis.reasoning_required,
            "entity_count": analysis.entity_count,
            "confidence": analysis.confidence,
            "reasoning": analysis.reasoning,
        }

    @staticmethod
    def sources_from_documents(documents: List) -> List[dict]:
        """从检索结果 Document 列表提取来源摘要信息（含 chunk 元信息与各通道得分，供前端可视化）。"""
        sources = []
        for doc in documents:
            md = doc.metadata
            sources.append({
                # 基础信息
                "recipe_name": md.get('recipe_name', '未知内容'),
                "search_type": md.get('search_type', md.get('route_strategy', 'unknown')),
                "search_method": md.get('search_method', md.get('search_type', 'unknown')),
                "score": float(md.get('final_score', md.get('relevance_score', 0))),
                "final_score": float(md.get('final_score', md.get('relevance_score', 0))) if md.get('final_score') is not None or md.get('relevance_score') is not None else None,
                "content_preview": doc.page_content.strip()[:300],
                # chunk / 文档定位（哪些 chunk 被检索到）
                "node_id": md.get('node_id'),
                "chunk_id": md.get('chunk_id'),
                "chunk_index": md.get('chunk_index'),
                "total_chunks": md.get('total_chunks'),
                "section_title": md.get('section_title'),
                # 各检索通道命中情况与得分对比
                "rrf_sources": md.get('rrf_sources'),
                "rrf_ranks": md.get('rrf_ranks'),
                "rrf_raw_scores": md.get('rrf_raw_scores'),
                "bm25_score": md.get('bm25_score'),
                "vector_score": md.get('score') if md.get('search_method') == 'vector' or 'rrf_sources' in md else None,
                "dual_score": md.get('relevance_score') if md.get('search_method') == 'dual_level' else None,
                # 重排元信息
                "rerank_score": md.get('rerank_score'),
                "reranked": md.get('reranked'),
                # 图 RAG 路径元信息
                "path_length": md.get('path_length'),
                "node_count": md.get('node_count'),
                "relationship_count": md.get('relationship_count'),
            })
        return sources

    def retrieval_trace_from_documents(self, documents: List, analysis) -> dict:
        """从检索结果构建「检索过程轨迹」，供前端展示为什么推荐这些结果。

        包含三部分（按策略自适应，缺失部分为 None / []）：
            - graph_query_plan: 图 RAG 查询规划（query_type / 源实体 / 目标实体 / 关系类型 / 最大跳数）
            - graph_paths: 图推理路径列表（节点链 + 关系链 + 跳数 + 相关性分），来自 graph_path / knowledge_subgraph 文档
            - channel_stats: 三路召回（dual_level / vector / bm25）候选与入选统计（仅混合检索有）

        Args:
            documents: 检索结果 Document 列表
            analysis: QueryAnalysis 对象（用于按策略决定读取哪部分，避免读到陈旧侧状态）

        Returns:
            {graph_query_plan, graph_paths, channel_stats}
        """
        strategy = analysis.recommended_strategy.value if analysis else None

        # ---- 图 RAG 部分：graph_rag / combined 时才有意义 ----
        graph_query_plan = None
        graph_paths: List[dict] = []
        if strategy in ("graph_rag", "combined") and self.graph_rag_retrieval is not None:
            graph_query_plan = getattr(self.graph_rag_retrieval, "last_query_plan", None)

        for doc in documents:
            md = doc.metadata
            st = md.get("search_type")
            if st not in ("graph_path", "knowledge_subgraph"):
                continue
            nodes = md.get("path_nodes") or []
            rels = md.get("path_relationships") or []
            graph_paths.append({
                "type": st,
                "recipe_name": md.get("recipe_name", "图结构结果"),
                "path_length": md.get("path_length"),
                "relevance_score": float(md["relevance_score"]) if md.get("relevance_score") is not None else None,
                "nodes": nodes,
                "relationships": rels,
                "node_count": md.get("node_count", len(nodes)),
                "relationship_count": md.get("relationship_count", len(rels)),
                "graph_density": md.get("graph_density"),
                "reasoning_chains": md.get("reasoning_chains") or [],
            })

        # ---- 通道统计部分：hybrid_traditional / combined 时才有意义 ----
        channel_stats = None
        rerank_stats = None
        if strategy in ("hybrid_traditional", "combined") and self.traditional_retrieval is not None:
            stats = getattr(self.traditional_retrieval, "last_hybrid_stats", None)
            if stats:
                # 统计最终结果中各通道实际入选数（按 rrf_sources 标记）
                contributed = {ch: 0 for ch in stats.get("channels", [])}
                for doc in documents:
                    for ch in (doc.metadata.get("rrf_sources") or []):
                        if ch in contributed:
                            contributed[ch] += 1
                channel_stats = {
                    "candidates": stats.get("candidates", {}),
                    "final": stats.get("final", len(documents)),
                    "channels": stats.get("channels", []),
                    "contributed": contributed,
                }
                # 重排统计（是否启用/生效、候选池大小等）
                rerank_stats = stats.get("rerank")

        return {
            "graph_query_plan": graph_query_plan,
            "graph_paths": graph_paths,
            "channel_stats": channel_stats,
            "rerank_stats": rerank_stats,
        }

    def get_system_stats(self) -> dict:
        """返回系统运行统计（路由占比 + 知识库统计 + Milvus 统计 + 就绪标记）。

        与 _show_system_stats / _show_knowledge_base_stats 的区别：本方法返回字典而非打印，
        供 Web API 直接序列化返回前端。
        """
        result = {"ready": self.system_ready}

        # 路由统计（各策略使用次数和占比）
        if self.query_router:
            result["route_stats"] = self.query_router.get_route_statistics()
        else:
            result["route_stats"] = {"total_queries": 0}

        # 知识库统计（菜谱/食材/步骤/文档/分块 + 分类）
        if self.data_module:
            result["knowledge_base"] = self.data_module.get_statistics()
        else:
            result["knowledge_base"] = {}

        # Milvus 统计（向量索引记录数）
        if self.index_module:
            result["milvus"] = self.index_module.get_collection_stats()
        else:
            result["milvus"] = {}

        return result

    def get_all_recipe_names(self) -> list:
        """返回所有菜谱的 id/name/category 列表（来自内存列表，免查库）。"""
        if not self.system_ready:
            raise ValueError("系统未就绪，请先构建知识库")
        return self.data_module.get_all_recipe_names()

    def get_recipe_list(self) -> list:
        """返回所有菜谱的完整列表（含难度、分类、食材数、步骤数等元数据）。"""
        if not self.system_ready:
            raise ValueError("系统未就绪，请先构建知识库")
        return self.data_module.get_recipe_list()

    def get_single_recipe_graph(self, recipe_id: str) -> dict:
        """返回指定菜谱的完整 1-hop 子图（所有食材/步骤/分类，无限制）。"""
        if not self.system_ready:
            raise ValueError("系统未就绪，请先构建知识库")
        return self.data_module.get_single_recipe_graph(recipe_id)

    def get_recipe_document(self, recipe_id: str) -> dict:
        """返回指定菜谱的完整文档内容（page_content + metadata），供前端文档详情展示。"""
        if not self.system_ready:
            raise ValueError("系统未就绪，请先构建知识库")
        for doc in self.data_module.documents:
            if doc.metadata.get("node_id") == recipe_id:
                return {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
        raise ValueError(f"未找到菜谱文档: {recipe_id}")

    def upload_markdown_recipe(self, content: str, filename: str = "") -> dict:
        """解析 Markdown 菜谱并写入知识库（Neo4j + Milvus + 内存索引）。

        供 Web API 调用，运行在线程池中（Neo4j 写入 + embedding 计算均为阻塞操作）。

        Args:
            content: Markdown 文件文本内容
            filename: 原始文件名（用于日志）

        Returns:
            {success, message, recipe_id, recipe_name, chunks_created, ingredients, steps, milvus_ok, stats}
        """
        if not self.system_ready:
            raise ValueError("系统未就绪，请先构建知识库")

        # 1. 解析 Markdown
        logger.info(f"开始解析上传的菜谱 Markdown: {filename or '(unnamed)'}")
        parsed = parse_markdown_recipe(content)
        logger.info(f"解析完成: 菜名='{parsed.name}', 食材{len(parsed.ingredients)}个, 步骤{len(parsed.steps)}个")

        # 2. 写入 Neo4j + 更新内存节点列表（含旧数据清理）
        ingest_result = self.data_module.ingest_markdown_recipe(parsed)
        recipe_id = ingest_result["recipe_id"]

        # 2.5 清理 Milvus 中旧上传菜谱的向量（覆盖上传场景）
        for old_id in ingest_result.get("old_deleted_ids", []):
            try:
                self.index_module.delete_by_node_id_prefix(old_id)
            except Exception as e:
                logger.warning(f"清理 Milvus 旧向量失败 ({old_id}): {e}")

        # 3. 构建 LangChain Document（与 build_recipe_documents 格式一致）
        content_parts = [f"# {parsed.name}"]
        if parsed.difficulty:
            content_parts.append(f"难度: {parsed.difficulty}星")
        if parsed.image_path:
            content_parts.append(f"图片: {parsed.image_path}")
        if parsed.ingredient_amounts:
            content_parts.append("\n## 所需食材")
            for i, ing in enumerate(parsed.ingredient_amounts, 1):
                line = f"{i}. {ing.name}"
                if ing.amount is not None and ing.unit:
                    line += f"({ing.amount}{ing.unit})"
                elif ing.unit:
                    line += f"({ing.unit})"
                content_parts.append(line)
        elif parsed.ingredients:
            content_parts.append("\n## 所需食材")
            for i, ing in enumerate(parsed.ingredients, 1):
                content_parts.append(f"{i}. {ing.name}")
        if parsed.steps:
            content_parts.append("\n## 制作步骤")
            for i, step in enumerate(parsed.steps, 1):
                content_parts.append(f"\n### 第{i}步\n步骤: {step}")
        if parsed.additional_notes:
            content_parts.append("\n## 附加内容")
            for note in parsed.additional_notes:
                content_parts.append(f"- {note}")

        full_content = "\n".join(content_parts)

        new_doc = Document(
            page_content=full_content,
            metadata={
                "node_id": recipe_id,
                "recipe_name": parsed.name,
                "node_type": "Recipe",
                "category": "用户上传",
                "cuisine_type": "用户上传",
                "difficulty": parsed.difficulty,
                "ingredients_count": len(parsed.ingredients),
                "steps_count": len(parsed.steps),
                "doc_type": "recipe",
                "content_length": len(full_content),
                "source": "markdown_upload",
            },
        )
        self.data_module.documents.append(new_doc)

        # 4. 对单篇文档分块（复用 chunk_documents 逻辑）
        new_chunks = []
        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap
        doc_content = new_doc.page_content

        if len(doc_content) <= chunk_size:
            chunk = Document(
                page_content=doc_content,
                metadata={
                    **new_doc.metadata,
                    "chunk_id": f"{recipe_id}_chunk_0",
                    "parent_id": recipe_id,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "chunk_size": len(doc_content),
                    "doc_type": "chunk",
                },
            )
            new_chunks.append(chunk)
        else:
            sections = doc_content.split('\n## ')
            if len(sections) <= 1:
                total_chunks = (len(doc_content) - 1) // (chunk_size - chunk_overlap) + 1
                for i in range(total_chunks):
                    start = i * (chunk_size - chunk_overlap)
                    end = min(start + chunk_size, len(doc_content))
                    chunk_content = doc_content[start:end]
                    new_chunks.append(Document(
                        page_content=chunk_content,
                        metadata={
                            **new_doc.metadata,
                            "chunk_id": f"{recipe_id}_chunk_{i}",
                            "parent_id": recipe_id,
                            "chunk_index": i,
                            "total_chunks": total_chunks,
                            "chunk_size": len(chunk_content),
                            "doc_type": "chunk",
                        },
                    ))
            else:
                total_chunks = len(sections)
                for i, section in enumerate(sections):
                    chunk_content = section if i == 0 else f"## {section}"
                    new_chunks.append(Document(
                        page_content=chunk_content,
                        metadata={
                            **new_doc.metadata,
                            "chunk_id": f"{recipe_id}_chunk_{i}",
                            "parent_id": recipe_id,
                            "chunk_index": i,
                            "total_chunks": total_chunks,
                            "chunk_size": len(chunk_content),
                            "doc_type": "chunk",
                            "section_title": section.split('\n')[0] if i > 0 else "主标题",
                        },
                    ))

        self.data_module.chunks.extend(new_chunks)

        # 5. 插入 Milvus（增量）
        milvus_ok = True
        try:
            milvus_ok = self.index_module.add_documents(new_chunks)
            if not milvus_ok:
                logger.warning("Milvus 插入失败，但 Neo4j 数据已保留")
        except Exception as e:
            logger.error(f"Milvus 插入异常: {e}")
            milvus_ok = False

        # 6. 重建 BM25 索引
        try:
            self.traditional_retrieval.rebuild_bm25()
            logger.info("BM25 索引重建完成")
        except Exception as e:
            logger.error(f"BM25 重建失败: {e}")

        # 7. 增量更新图 KV 索引
        try:
            self.traditional_retrieval.add_recipe_to_graph_index(
                recipe_node=self.data_module.recipes[-1],  # 刚添加的 recipe GraphNode
                parsed=parsed,
                recipe_id=recipe_id,
                ingredient_ids=ingest_result["ingredient_ids"],
            )
            logger.info("图 KV 索引增量更新完成")
        except Exception as e:
            logger.error(f"图 KV 索引更新失败: {e}")

        # 8. 更新父文档映射
        try:
            self.traditional_retrieval._parent_doc_map[recipe_id] = new_doc
        except Exception:
            pass

        # 9. 清除文档缓存（重启时从 Neo4j 重新加载）
        self.data_module.clear_documents_cache()

        message = f"菜谱 '{parsed.name}' 上传成功"
        if not milvus_ok:
            message += "（向量索引更新失败，可稍后重建知识库修复）"

        return {
            "success": True,
            "message": message,
            "recipe_id": recipe_id,
            "recipe_name": parsed.name,
            "chunks_created": len(new_chunks),
            "ingredients": len(parsed.ingredients),
            "steps": len(parsed.steps),
            "milvus_ok": milvus_ok,
            "stats": self.get_system_stats(),
        }

    def rebuild_knowledge_base(self) -> dict:
        """重建知识库（删除现有向量数据并重新构建），无确认 prompt。

        与 _rebuild_knowledge_base 的区别：本方法不做 input() 确认，
        供 Web API 调用（确认逻辑由前端 / API 层负责）。

        Returns:
            {"success": bool, "message": str, "stats": dict}
        """
        try:
            # 删除现有的 Milvus 集合
            if self.index_module.has_collection():
                self.index_module.delete_collection()

            # 清理菜谱文档缓存，确保从 Neo4j 重新构建
            self.data_module.clear_documents_cache()

            # 重新构建知识库
            self.build_knowledge_base()

            return {
                "success": True,
                "message": "知识库重建完成",
                "stats": self.get_system_stats(),
            }
        except Exception as e:
            logger.error(f"重建知识库失败: {e}")
            return {
                "success": False,
                "message": f"重建失败：{e}",
                "stats": self.get_system_stats(),
            }

    def run_interactive(self):
        """运行交互式问答循环。

        提供以下命令：
            - 'stats'   : 查看系统统计（路由占比 + 知识库统计）
            - 'rebuild' : 重建知识库（删除现有向量数据并重新构建）
            - 'quit'    : 退出系统
            - 其他输入 : 作为问题进行智能问答（默认流式输出）

        Note:
            系统未就绪时会提示用户先构建知识库。
        """
        if not self.system_ready:
            print("❌ 系统未就绪，请先构建知识库")
            return

        print("\n欢迎使用尝尝咸淡RAG烹饪助手！")
        print("可用功能：")
        print("   - 'stats' : 查看系统统计")
        print("   - 'rebuild' : 重建知识库")
        print("   - 'quit' : 退出系统")
        print("\n" + "="*50)

        while True:
            try:
                user_input = input("\n您的问题: ").strip()

                if not user_input:
                    continue

                # 命令处理
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'stats':
                    self._show_system_stats()
                    continue
                elif user_input.lower() == 'rebuild':
                    self._rebuild_knowledge_base()
                    continue

                # 普通问答 - 使用默认设置
                use_stream = True          # 默认使用流式输出（提升交互体验）
                explain_routing = False    # 默认不显示路由决策（减少输出噪声）

                print("\n回答:")

                result, analysis = self.ask_question_with_routing(
                    user_input,
                    stream=use_stream,
                    explain_routing=explain_routing
                )

                # 非流式模式下打印完整结果（流式已在 ask_question_with_routing 中实时输出）
                if not use_stream and result:
                    print(f"{result}\n")

            except KeyboardInterrupt:
                # Ctrl+C 退出循环
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
                import traceback
                traceback.print_exc()

        print("\n👋 感谢使用尝尝咸淡RAG烹饪助手！")
        self._cleanup()

    def _show_system_stats(self):
        """显示系统运行统计（路由占比 + 知识库统计）。

        路由统计展示各检索策略的使用次数和占比，
        知识库统计展示节点数、向量数、分类等信息。
        """
        print("\n系统运行统计")
        print("=" * 40)

        # 路由统计（各策略使用次数和占比）
        route_stats = self.query_router.get_route_statistics()
        total_queries = route_stats.get('total_queries', 0)

        if total_queries > 0:
            print(f"总查询次数: {total_queries}")
            print(f"传统检索: {route_stats.get('traditional_count', 0)} ({route_stats.get('traditional_ratio', 0):.1%})")
            print(f"图RAG检索: {route_stats.get('graph_rag_count', 0)} ({route_stats.get('graph_rag_ratio', 0):.1%})")
            print(f"组合策略: {route_stats.get('combined_count', 0)} ({route_stats.get('combined_ratio', 0):.1%})")
        else:
            print("暂无查询记录")

        # 知识库统计（复用 _show_knowledge_base_stats）
        self._show_knowledge_base_stats()

    def _rebuild_knowledge_base(self):
        """重建知识库（删除现有向量数据并重新构建）。

        执行流程：
            1. 用户确认（防止误操作）
            2. 删除现有 Milvus 集合
            3. 调用 build_knowledge_base 重新构建

        Note:
            重建会清空所有向量数据，需重新从 Neo4j 加载并构建索引。
        """
        print("\n准备重建知识库...")

        # 确认操作（防止误删除）
        confirm = input("⚠️  这将删除现有的向量数据并重新构建，是否继续？(y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ 重建操作已取消")
            return

        # 复用 Web API 共享的重建逻辑（无确认 prompt 版本）
        result = self.rebuild_knowledge_base()
        if result["success"]:
            print("✅ 知识库重建完成！")
        else:
            print(f"❌ 重建失败: {result['message']}")
            print("建议：请检查Milvus服务状态后重试")

    def _cleanup(self):
        """清理资源：关闭各模块的数据库连接。

        按依赖关系逆序关闭：数据模块 -> 传统检索 -> 图 RAG 检索 -> 向量索引。
        """
        if self.data_module:
            self.data_module.close()
        if self.traditional_retrieval:
            self.traditional_retrieval.close()
        if self.graph_rag_retrieval:
            self.graph_rag_retrieval.close()
        if self.index_module:
            self.index_module.close()

def main():
    """主函数：系统启动入口。

    执行流程：
        1. 创建图 RAG 系统实例
        2. 初始化系统（实例化 6 大核心模块）
        3. 构建知识库（加载或新建）
        4. 运行交互式问答循环

    异常处理：
        任何阶段失败时记录日志并打印堆栈，便于排查问题。
    """
    try:
        print("启动高级图RAG系统...")

        # 创建高级图 RAG 系统（使用默认配置）
        rag_system = AdvancedGraphRAGSystem()

        # 初始化系统（实例化各核心模块）
        rag_system.initialize_system()

        # 构建知识库（加载已存在或从 Neo4j 新建）
        rag_system.build_knowledge_base()

        # 运行交互式问答循环
        rag_system.run_interactive()

    except Exception as e:
        logger.error(f"系统运行失败: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ 系统错误: {e}")

if __name__ == "__main__":
    main()