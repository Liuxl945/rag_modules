import { defineStore } from 'pinia'
import { ref, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import { streamQuery } from '@/utils/sse'
import {
  listConversations,
  createConversation,
  getConversation,
  appendMessage,
  renameConversation as renameConversationApi,
  deleteConversation as deleteConversationApi,
} from '@/api'
import type { ChatMessage, ConversationMeta } from '@/types'

/**
 * 会话 store - 统管多会话列表、当前会话消息与流式生成逻辑。
 *
 * ChatView 只负责展示与 DOM 滚动；所有「发消息 / 切换 / 新建 / 重命名 / 删除」
 * 状态变更都走这里，保证侧边栏与聊天区数据一致。
 */
export const useConversationStore = defineStore('conversations', () => {
  /** 侧边栏会话列表（元数据，按 updated_at 倒序） */
  const conversations = ref<ConversationMeta[]>([])
  /** 当前会话 id；null 表示「新对话」未发送，聊天区显示空态建议 */
  const activeId = ref<string | null>(null)
  /** 当前会话的消息（展示用） */
  const messages = ref<ChatMessage[]>([])
  /** 流式生成中 */
  const loading = ref(false)
  /** 首次列表是否已加载（区分「加载中」与「真的没有会话」） */
  const loaded = ref(false)

  let abortController: AbortController | null = null
  let localIdSeq = 0

  function nextLocalId(): string {
    // 前缀 local- 避免与后端 msg-N 冲突；切换会话从后端重新加载时用真实 id
    localIdSeq += 1
    return `local-${localIdSeq}`
  }

  /** 后端消息 -> 前端展示消息 */
  function normalizeMsg(m: any): ChatMessage {
    return {
      id: m.id,
      role: m.role,
      content: m.content || '',
      analysis: m.analysis,
      sources: m.sources,
      retrieval_trace: m.retrieval_trace,
      elapsed: m.elapsed,
      streaming: false,
      error: m.error || false,
    }
  }

  /** 从完整会话提取元数据（用于列表） */
  function metaFromConversation(c: any): ConversationMeta {
    return {
      id: c.id,
      title: c.title,
      created_at: c.created_at,
      updated_at: c.updated_at,
      message_count: (c.messages || []).length,
      last_message_preview:
        (c.messages || []).filter((m: any) => m.content).slice(-1)[0]?.content?.slice(0, 60) || '',
    }
  }

  /** 更新或插入会话元数据，并按 updated_at 倒序重排 */
  function upsertMeta(meta: ConversationMeta) {
    const idx = conversations.value.findIndex((c) => c.id === meta.id)
    if (idx >= 0) conversations.value[idx] = meta
    else conversations.value.unshift(meta)
    conversations.value.sort((a, b) => (b.updated_at || 0) - (a.updated_at || 0))
  }

  /** 异步持久化一条消息到后端，成功后用返回的元数据刷新侧边栏 */
  function persistMessage(convId: string, msg: ChatMessage) {
    appendMessage(convId, {
      role: msg.role,
      content: msg.content,
      analysis: msg.analysis,
      sources: msg.sources,
      retrieval_trace: msg.retrieval_trace,
      elapsed: msg.elapsed,
      error: msg.error || false,
      timestamp: Date.now() / 1000,
    })
      .then((res) => upsertMeta(res.conversation))
      .catch((err) => console.warn('保存消息失败', err))
  }

  /** 中止当前流式生成；persist=true 时把已生成的部分回答落盘 */
  function abortStream(persist = true) {
    // 无论是否有进行中的流，都先解除 loading（stop 也复用此函数）
    loading.value = false
    if (abortController) {
      abortController.abort()
      abortController = null
    }
    const last = messages.value[messages.value.length - 1]
    if (last && last.role === 'assistant' && last.streaming) {
      last.streaming = false
      if (!last.content) last.content = '（已停止生成）'
      if (persist && activeId.value) persistMessage(activeId.value, last)
    }
  }

  // ------------------------------------------------------------------
  // 列表 / 切换 / 新建
  // ------------------------------------------------------------------
  async function fetchConversations() {
    try {
      const list = await listConversations()
      conversations.value = list
      // 有历史且未选中任何会话 -> 自动打开最近一条
      if (list.length && !activeId.value) {
        await selectConversation(list[0].id)
      }
    } catch (err) {
      console.warn('加载会话列表失败', err)
    } finally {
      loaded.value = true
    }
  }

  async function selectConversation(id: string) {
    // 已选中且已加载 -> 不重复请求
    if (id === activeId.value && messages.value.length) return
    abortStream() // 切走前先中止并保存当前生成
    activeId.value = id
    messages.value = []
    try {
      const conv = await getConversation(id)
      // 请求期间用户可能又切了会话，校验后再赋值
      if (activeId.value !== id) return
      messages.value = (conv.messages || []).map(normalizeMsg)
    } catch (err) {
      console.warn('加载会话失败', err)
      ElMessage.error('加载会话失败')
    }
  }

  function startNewConversation() {
    abortStream()
    activeId.value = null
    messages.value = []
  }

  /** 若当前没有活动会话，用首条问题作标题新建一个；返回活动会话 id */
  async function ensureConversation(firstQuestion: string): Promise<string | null> {
    if (activeId.value) return activeId.value
    try {
      const conv = await createConversation(firstQuestion.slice(0, 24))
      upsertMeta(metaFromConversation(conv))
      activeId.value = conv.id
      return conv.id
    } catch (err) {
      console.warn('创建会话失败', err)
      ElMessage.error('创建会话失败')
      return null
    }
  }

  // ------------------------------------------------------------------
  // 发送 / 停止
  // ------------------------------------------------------------------
  async function sendMessage(question: string) {
    if (loading.value) return
    const q = question.trim()
    if (!q) return

    // 早置 loading：阻塞输入框与建议按钮的重复触发（ensureConversation 是异步的）
    loading.value = true

    const convId = await ensureConversation(q)
    if (!convId) {
      loading.value = false
      return
    }
    // 用户在创建会话期间点了停止 -> 放弃本次发送（已建的空会话保留，可稍后继续）
    if (!loading.value) return

    // 1. 用户消息：先本地展示，再异步落盘
    const userMsg: ChatMessage = { id: nextLocalId(), role: 'user', content: q }
    messages.value.push(userMsg)
    persistMessage(convId, userMsg)

    // 2. 助手占位消息（reactive：push 后仍通过同一 proxy 修改，回调里的属性赋值才能触发视图更新）
    const assistantMsg = reactive<ChatMessage>({
      id: nextLocalId(),
      role: 'assistant',
      content: '',
      streaming: true,
    })
    messages.value.push(assistantMsg)

    abortController = new AbortController()

    await streamQuery(
      q,
      null,
      {
        onAnalysis: ({ analysis, sources, retrieval_trace }) => {
          assistantMsg.analysis = analysis
          assistantMsg.sources = sources
          assistantMsg.retrieval_trace = retrieval_trace ?? null
        },
        onChunk: ({ content }) => {
          assistantMsg.content += content
        },
        onDone: ({ elapsed }) => {
          assistantMsg.streaming = false
          if (elapsed != null) assistantMsg.elapsed = elapsed
          loading.value = false
          abortController = null
          persistMessage(convId, assistantMsg)
        },
        onError: ({ message }) => {
          assistantMsg.streaming = false
          assistantMsg.error = true
          assistantMsg.content = assistantMsg.content
            ? `${assistantMsg.content}\n\n⚠️ ${message}`
            : `⚠️ ${message}`
          loading.value = false
          abortController = null
          persistMessage(convId, assistantMsg)
        },
      },
      abortController.signal,
    )

    // 安全网：流提前结束但未触发 done/error（如 abort 后 streamQuery 静默返回）
    if (loading.value) {
      assistantMsg.streaming = false
      loading.value = false
      abortController = null
      persistMessage(convId, assistantMsg)
    }
  }

  function stop() {
    abortStream(true)
  }

  // ------------------------------------------------------------------
  // 重命名 / 删除
  // ------------------------------------------------------------------
  async function renameConversation(id: string, title: string): Promise<boolean> {
    try {
      const meta = await renameConversationApi(id, title)
      upsertMeta(meta)
      return true
    } catch (err) {
      console.warn('重命名会话失败', err)
      ElMessage.error('重命名失败')
      return false
    }
  }

  async function deleteConversation(id: string) {
    // 删除的是当前会话时，先中止生成（不落盘，反正要删）
    if (activeId.value === id) abortStream(false)
    try {
      await deleteConversationApi(id)
    } catch (err) {
      console.warn('删除会话失败', err)
      ElMessage.error('删除失败')
      return
    }
    const idx = conversations.value.findIndex((c) => c.id === id)
    if (idx >= 0) conversations.value.splice(idx, 1)

    if (activeId.value === id) {
      // 删除的是当前会话：切到相邻会话，没有则进入新对话空态
      const next = conversations.value[idx] || conversations.value[idx - 1] || null
      if (next) await selectConversation(next.id)
      else startNewConversation()
    }
  }

  return {
    conversations,
    activeId,
    messages,
    loading,
    loaded,
    fetchConversations,
    selectConversation,
    startNewConversation,
    sendMessage,
    stop,
    renameConversation,
    deleteConversation,
  }
})
