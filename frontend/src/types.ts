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

/** 检索来源摘要 */
export interface SourceDoc {
  recipe_name: string
  search_type: string
  score: number
  content_preview: string
}

/** 非流式问答响应 */
export interface QueryResponse {
  answer: string
  analysis: AnalysisInfo
  sources: SourceDoc[]
  elapsed: number
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
