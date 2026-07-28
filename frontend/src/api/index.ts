import axios from 'axios'
import type {
  HealthResponse,
  QueryResponse,
  RebuildResponse,
  SystemStats,
  ConversationMeta,
  Conversation,
  ChatMessage,
  KnowledgeGraph,
  RecipeName,
  RecipeListItem,
  RecipeDocument,
  UploadRecipeResponse,
  EvaluationDatasetItem,
  EvaluationRunResult,
  EvaluationSingleResponse,
  MessageEvaluationResponse,
  EvaluationRunSummary,
  EvaluationStatus,
} from '@/types'

const http = axios.create({
  baseURL: '/api',
  timeout: 120000, // 非流式问答可能较慢，给 2 分钟
})

/** 健康检查（系统就绪状态） */
export async function getHealth(): Promise<HealthResponse> {
  const { data } = await http.get<HealthResponse>('/health')
  return data
}

/** 系统统计 */
export async function getStats(): Promise<SystemStats> {
  const { data } = await http.get<SystemStats>('/stats')
  return data
}

/** 获取所有菜谱的 id/name/category 列表（下拉选择用） */
export async function listRecipeNames(): Promise<RecipeName[]> {
  const { data } = await http.get<{ recipes: RecipeName[] }>('/recipes')
  return data.recipes
}

/** 获取所有菜谱的完整列表（含难度、分类、食材数等元数据，浏览页用） */
export async function listRecipesFull(): Promise<RecipeListItem[]> {
  const { data } = await http.get<{ recipes: RecipeListItem[] }>('/recipes/list')
  return data.recipes
}

/** 获取指定菜谱的完整 1-hop 子图（所有食材/步骤/分类，无限制） */
export async function getRecipeGraph(recipeId: string): Promise<KnowledgeGraph> {
  const { data } = await http.get<KnowledgeGraph>(
    `/knowledge-graph/recipe/${encodeURIComponent(recipeId)}`,
  )
  return data
}

/** 获取指定菜谱的完整文档内容（markdown 文本 + 元数据） */
export async function getRecipeDocument(recipeId: string): Promise<RecipeDocument> {
  const { data } = await http.get<RecipeDocument>(
    `/recipe-document/${encodeURIComponent(recipeId)}`,
  )
  return data
}

/** 上传 Markdown 菜谱文件 */
export async function uploadRecipe(file: File): Promise<UploadRecipeResponse> {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await http.post<UploadRecipeResponse>('/recipes/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 60000,
  })
  return data
}

/** 非流式问答 */
export async function query(question: string, topK: number | null = null): Promise<QueryResponse> {
  const { data } = await http.post<QueryResponse>('/query', { question, top_k: topK })
  return data
}

/** 重建知识库 */
export async function rebuildKnowledgeBase(): Promise<RebuildResponse> {
  const { data } = await http.post<RebuildResponse>('/knowledge-base/rebuild')
  return data
}

// ---------------------------------------------------------------------------
// 聊天会话（多会话历史）
// ---------------------------------------------------------------------------

/** 列出所有会话元数据（按最近更新倒序） */
export async function listConversations(): Promise<ConversationMeta[]> {
  const { data } = await http.get<{ conversations: ConversationMeta[] }>('/conversations')
  return data.conversations
}

/** 新建空会话，返回完整会话 */
export async function createConversation(title?: string): Promise<Conversation> {
  const { data } = await http.post<{ conversation: Conversation }>('/conversations', { title })
  return data.conversation
}

/** 获取指定会话完整内容（含消息） */
export async function getConversation(id: string): Promise<Conversation> {
  const { data } = await http.get<{ conversation: Conversation }>(`/conversations/${id}`)
  return data.conversation
}

/** 向会话追加一条消息，返回该消息与更新后的会话元数据 */
export async function appendMessage(
  id: string,
  payload: {
    role: string
    content: string
    analysis?: any
    sources?: any
    retrieval_trace?: any
    elapsed?: number
    error?: boolean
    timestamp?: number
  },
): Promise<{ message: ChatMessage; conversation: ConversationMeta }> {
  const { data } = await http.post<{ message: ChatMessage; conversation: ConversationMeta }>(
    `/conversations/${id}/messages`,
    payload,
  )
  return data
}

/** 重命名会话，返回更新后的元数据 */
export async function renameConversation(id: string, title: string): Promise<ConversationMeta> {
  const { data } = await http.patch<{ conversation: ConversationMeta }>(`/conversations/${id}`, {
    title,
  })
  return data.conversation
}

/** 删除会话 */
export async function deleteConversation(id: string): Promise<void> {
  await http.delete(`/conversations/${id}`)
}

// ---------------------------------------------------------------------------
// RAGAS 评估
// ---------------------------------------------------------------------------

/** 评估依赖可用性（RAGAS 是否已安装） */
export async function getEvaluationStatus(): Promise<EvaluationStatus> {
  const { data } = await http.get<EvaluationStatus>('/evaluation/status')
  return data
}

/** 内置烹饪评估测试集 */
export async function getEvaluationDataset(): Promise<EvaluationDatasetItem[]> {
  const { data } = await http.get<{ dataset: EvaluationDatasetItem[] }>('/evaluation/dataset')
  return data.dataset
}

/** 手动单条评估：question / answer / contexts / ground_truth(可选) */
export async function evaluateSingle(payload: {
  question: string
  answer: string
  contexts?: string[]
  ground_truth?: string
}): Promise<EvaluationSingleResponse> {
  const { data } = await http.post<EvaluationSingleResponse>('/evaluation/single', payload, {
    timeout: 300000, // 5 分钟
  })
  return data
}

/** 评估会话中的某条助手问答（重新检索取完整 contexts） */
export async function evaluateMessage(
  conversationId: string,
  messageId: string,
): Promise<MessageEvaluationResponse> {
  const { data } = await http.post<MessageEvaluationResponse>(
    '/evaluation/message',
    { conversation_id: conversationId, message_id: messageId },
    { timeout: 300000 },
  )
  return data
}

/** 运行测试集评估（重，可能数分钟）；questions 留空用内置测试集 */
export async function runEvaluation(questions?: string[]): Promise<EvaluationRunResult> {
  const { data } = await http.post<EvaluationRunResult>(
    '/evaluation/run',
    { questions },
    { timeout: 900000 }, // 15 分钟：RAGAS 串行/低并发评估 + judge 重试，留足余量
  )
  return data
}

/** 历史评估运行列表（元数据） */
export async function listEvaluationResults(): Promise<EvaluationRunSummary[]> {
  const { data } = await http.get<{ results: EvaluationRunSummary[] }>('/evaluation/results')
  return data.results
}

/** 单次评估运行详情（含每样本 results） */
export async function getEvaluationResult(runId: string): Promise<EvaluationRunResult> {
  const { data } = await http.get<{ result: EvaluationRunResult }>(
    `/evaluation/results/${runId}`,
  )
  return data.result
}

/** 删除一次评估记录 */
export async function deleteEvaluationResult(runId: string): Promise<void> {
  await http.delete(`/evaluation/results/${runId}`)
}
