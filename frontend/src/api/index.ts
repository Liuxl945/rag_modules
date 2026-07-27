import axios from 'axios'
import type { HealthResponse, QueryResponse, RebuildResponse, SystemStats } from '@/types'

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
