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
