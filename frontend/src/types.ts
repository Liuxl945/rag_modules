// 与后端 backend/api/schemas.py 对应的前端类型定义

/** 路由分析结果 */
export interface AnalysisInfo {
  recommended_strategy: string // hybrid_traditional | graph_rag | combined
  query_complexity: number
  relationship_intensity: number
  reasoning_required: boolean
  entity_count: number
  confidence: number
  reasoning: string
}

/** 检索来源摘要（含 chunk 元信息与各通道得分） */
export interface SourceDoc {
  // 基础信息
  recipe_name: string
  search_type: string
  search_method?: string | null
  score: number
  final_score?: number | null
  content_preview: string
  // chunk / 文档定位
  node_id?: string | null
  chunk_id?: string | null
  chunk_index?: number | null
  total_chunks?: number | null
  section_title?: string | null
  // 各检索通道命中情况与得分对比
  rrf_sources?: string[] | null
  rrf_ranks?: Record<string, number> | null
  rrf_raw_scores?: Record<string, number> | null
  bm25_score?: number | null
  vector_score?: number | null
  dual_score?: number | null
  // 图 RAG 路径元信息
  path_length?: number | null
  node_count?: number | null
  relationship_count?: number | null
}

/** 图 RAG 查询规划（走了哪些实体/关系、最大跳数） */
export interface GraphQueryPlan {
  query_type: string
  source_entities: string[]
  target_entities: string[]
  relation_types: string[]
  max_depth: number
}

/** 图路径上的节点 */
export interface GraphPathNode {
  name: string
  labels?: string[] | null
}

/** 图路径上的关系 */
export interface GraphPathRel {
  type: string
}

/** 图推理路径（节点链 + 关系链 + 跳数 + 相关性分） */
export interface GraphPath {
  type: string // graph_path | knowledge_subgraph
  recipe_name: string
  path_length?: number | null
  relevance_score?: number | null
  nodes: GraphPathNode[]
  relationships: GraphPathRel[]
  node_count?: number | null
  relationship_count?: number | null
  graph_density?: number | null
  reasoning_chains: string[]
}

/** 三路召回（dual_level / vector / bm25）统计 */
export interface ChannelStats {
  candidates: Record<string, number>
  final: number
  channels: string[]
  contributed: Record<string, number>
}

/** 检索过程轨迹：为什么推荐这些结果 */
export interface RetrievalTrace {
  graph_query_plan?: GraphQueryPlan | null
  graph_paths: GraphPath[]
  channel_stats?: ChannelStats | null
}

/** 非流式问答响应 */
export interface QueryResponse {
  answer: string
  analysis: AnalysisInfo
  sources: SourceDoc[]
  elapsed: number
  retrieval_trace?: RetrievalTrace | null
}

/** 健康检查响应 */
export interface HealthResponse {
  ready: boolean
  status: 'initializing' | 'ready' | 'error'
  message: string | null
}

/** 重建知识库响应 */
export interface RebuildResponse {
  success: boolean
  message: string
  stats: SystemStats
}

/** 系统统计 */
export interface SystemStats {
  ready: boolean
  route_stats: {
    total_queries: number
    traditional_count?: number
    graph_rag_count?: number
    combined_count?: number
    traditional_ratio?: number
    graph_rag_ratio?: number
    combined_ratio?: number
  }
  knowledge_base: Record<string, number | Record<string, number>>
  milvus: Record<string, number>
}

/** 聊天消息（前端展示用） */
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  analysis?: AnalysisInfo
  sources?: SourceDoc[]
  retrieval_trace?: RetrievalTrace | null
  elapsed?: number
  streaming?: boolean
  error?: boolean
}

/** 会话元数据（侧边栏列表项，不含消息全文） */
export interface ConversationMeta {
  id: string
  title: string
  created_at: number
  updated_at: number
  message_count: number
  last_message_preview?: string
}

/** 完整会话（含消息列表） */
export interface Conversation extends ConversationMeta {
  messages: ChatMessage[]
}

/** 知识图谱节点（可视化用） */
export interface KnowledgeNode {
  id: string
  label: string
  type: 'Recipe' | 'Ingredient' | 'CookingStep' | 'Category' | string
  properties?: Record<string, any>
}

/** 知识图谱边（关系） */
export interface KnowledgeEdge {
  from: string
  to: string
  type: string
}

/** 知识子图（节点 + 边 + 计数） */
export interface KnowledgeGraph {
  nodes: KnowledgeNode[]
  edges: KnowledgeEdge[]
  counts: { primary: number; total: number }
}

/** 菜谱名称与分类（下拉选择用） */
export interface RecipeName {
  id: string
  name: string
  category: string
}

/** 菜谱完整信息（浏览页列表用） */
export interface RecipeListItem {
  id: string
  name: string
  category: string
  cuisine_type: string
  difficulty: number
  description: string
  image_path: string
  source: string
  ingredients_count: number
  steps_count: number
}

/** 菜谱文档详情（完整文档内容 + 元数据） */
export interface RecipeDocument {
  content: string
  metadata: Record<string, any>
}

/** Markdown 菜谱上传响应 */
export interface UploadRecipeResponse {
  success: boolean
  message: string
  recipe_id?: string
  recipe_name?: string
  chunks_created?: number
  ingredients?: number
  steps?: number
  milvus_ok?: boolean
  stats?: SystemStats
}

// ---------------------------------------------------------------------------
// RAGAS 评估
// ---------------------------------------------------------------------------

/** 内置测试集条目 */
export interface EvaluationDatasetItem {
  id: string
  question: string
  ground_truth: string
  category?: string | null
}

/** 指标列名 -> 得分（null 表示该指标未跑或单样本异常） */
export type EvaluationScores = Record<string, number | null>

/** 单样本评估结果（question + 各指标得分） */
export interface EvaluationSampleResult {
  question: string
  [metric: string]: string | number | null
}

/** 测试集评估运行完整结果 */
export interface EvaluationRunResult {
  run_id: string
  results: EvaluationSampleResult[]
  aggregates: EvaluationScores
  metrics: string[]
  count: number
  elapsed?: number | null
  skipped: number
}

/** 单条评估响应 */
export interface EvaluationSingleResponse {
  scores: EvaluationScores
  metrics: string[]
  count: number
  elapsed?: number | null
}

/** 会话消息评估响应（含回填的问题） */
export interface MessageEvaluationResponse extends EvaluationSingleResponse {
  question: string
}

/** 评估运行元数据（列表项，不含 results 全文） */
export interface EvaluationRunSummary {
  id: string
  created_at: number
  kind: string
  count: number
  metrics: string[]
  aggregates: EvaluationScores
  elapsed?: number | null
}

/** 评估依赖可用性 */
export interface EvaluationStatus {
  available: boolean
}
