import type { AnalysisInfo, SourceDoc, RetrievalTrace } from '@/types'

/**
 * SSE 流式问答客户端
 *
 * 浏览器原生 EventSource 只支持 GET，无法发 POST body，
 * 因此用 fetch + ReadableStream 手动解析 text/event-stream。
 *
 * 事件序列（与后端 backend/api/routes.py 一致）：
 *   analysis -> { analysis, sources, retrieval_trace }   路由检索完成
 *   chunk    -> { content }             逐 token（0..N 次）
 *   done     -> { elapsed?, answer? }   正常结束
 *   error    -> { message }             出错
 */

export interface StreamHandlers {
  onAnalysis?: (data: { analysis: AnalysisInfo; sources: SourceDoc[]; retrieval_trace?: RetrievalTrace | null }) => void
  onChunk?: (data: { content: string }) => void
  onDone?: (data: { elapsed?: number; answer?: string }) => void
  onError?: (data: { message: string }) => void
}

/** 解析单个 SSE 事件块（形如 "event: chunk\ndata: {...}"） */
function parseSSE(raw: string): { event: string; data: string } | null {
  let event = 'message'
  const dataLines: string[] = []
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      // sse-starlette 的 data: 后可能有一个前导空格
      dataLines.push(line.slice(5).replace(/^ /, ''))
    }
    // 以 : 开头的注释行（心跳）忽略
  }
  if (dataLines.length === 0) return null
  return { event, data: dataLines.join('\n') }
}

/**
 * 发起流式问答请求。
 * @param question 用户问题
 * @param topK 返回结果数量上限
 * @param handlers 事件回调
 * @param signal AbortSignal，用于「停止生成」
 */
export async function streamQuery(
  question: string,
  topK: number | null,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  let resp: Response
  try {
    resp = await fetch('/api/query/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k: topK }),
      signal,
    })
  } catch (err) {
    // 用户主动 abort 时 fetch 抛 AbortError，不计为错误
    if (err instanceof DOMException && err.name === 'AbortError') return
    handlers.onError?.({ message: '网络请求失败，请检查后端是否启动' })
    return
  }

  if (!resp.ok || !resp.body) {
    let message = `请求失败（${resp.status}）`
    try {
      const err = await resp.json()
      message = err.detail || message
    } catch {
      /* ignore */
    }
    handlers.onError?.({ message })
    return
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      // sse-starlette 用 CRLF(\r\n) 作行结束符，事件间为 \r\n\r\n；
      // 统一成 \n，否则下方 indexOf('\n\n') 永远匹配不到，事件全部丢失
      buffer = buffer.replace(/\r\n?/g, '\n')

      // SSE 事件以空行（\n\n）分隔，可能一次读到多个完整事件
      let sep: number
      while ((sep = buffer.indexOf('\n\n')) >= 0) {
        const raw = buffer.slice(0, sep)
        buffer = buffer.slice(sep + 2)
        const evt = parseSSE(raw)
        if (!evt) continue
        try {
          const payload = JSON.parse(evt.data)
          switch (evt.event) {
            case 'analysis':
              handlers.onAnalysis?.(payload)
              break
            case 'chunk':
              handlers.onChunk?.(payload)
              break
            case 'done':
              handlers.onDone?.(payload)
              break
            case 'error':
              handlers.onError?.(payload)
              break
          }
        } catch {
          /* 单条事件解析失败则跳过，不影响整体流 */
        }
      }
    }
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') return
    handlers.onError?.({ message: '流式读取中断' })
  }
}
