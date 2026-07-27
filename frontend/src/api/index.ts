import axios from 'axios'
import type {
  HealthResponse,
  QueryResponse,
  RebuildResponse,
  SystemStats,
  ConversationMeta,
  Conversation,
  ChatMessage,
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
