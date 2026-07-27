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
    GenerationIntegrationModule
)
from rag_modules.hybrid_retrieval import HybridRetrievalModule
from rag_modules.graph_rag_retrieval import GraphRAGRetrieval
from rag_modules.intelligent_query_router import IntelligentQueryRouter, QueryAnalysis

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
        """从检索结果 Document 列表提取来源摘要信息。"""
        sources = []
        for doc in documents:
            sources.append({
                "recipe_name": doc.metadata.get('recipe_name', '未知内容'),
                "search_type": doc.metadata.get('search_type', doc.metadata.get('route_strategy', 'unknown')),
                "score": float(doc.metadata.get('final_score', doc.metadata.get('relevance_score', 0))),
                "content_preview": doc.page_content.strip()[:200],
            })
        return sources

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

    def get_knowledge_subgraph(self, node_type: str, limit: int = 15, neighbor_limit: int = 6) -> dict:
        """返回某类节点（recipes/ingredients/cooking_steps）的有界知识子图，供前端可视化。

        透传给 data_module.get_knowledge_subgraph；系统未就绪时抛 ValueError。
        """
        if not self.system_ready:
            raise ValueError("系统未就绪，请先构建知识库")
        return self.data_module.get_knowledge_subgraph(node_type, limit, neighbor_limit)

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