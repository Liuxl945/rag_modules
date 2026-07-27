<script setup lang="ts">
import { ref, nextTick, onMounted, reactive } from 'vue'
import ChatMessage from '@/components/ChatMessage.vue'
import MessageInput from '@/components/MessageInput.vue'
import { streamQuery } from '@/utils/sse'
import { useSystemStore } from '@/stores/system'
import type { ChatMessage as ChatMessageType } from '@/types'

const systemStore = useSystemStore()

const messages = ref<ChatMessageType[]>([])
const loading = ref(false)
const scrollRef = ref<HTMLDivElement | null>(null)
let abortController: AbortController | null = null
let idSeq = 0

function nextId(): string {
  idSeq += 1
  return `msg-${idSeq}`
}

async function scrollToBottom() {
  await nextTick()
  const el = scrollRef.value
  if (el) el.scrollTop = el.scrollHeight
}

async function send(question: string) {
  if (loading.value) return

  // 用户消息
  messages.value.push({ id: nextId(), role: 'user', content: question })

  // 占位的助手消息（流式填充）
  // 用 reactive 包裹：push 后仍通过同一 proxy 修改，回调里的属性赋值才能触发视图更新。
  // 若用普通对象，push 进 ref 数组后 assistantMsg 仍指向原始对象，改属性不会更新视图。
  const assistantMsg = reactive<ChatMessageType>({
    id: nextId(),
    role: 'assistant',
    content: '',
    streaming: true,
  })
  messages.value.push(assistantMsg)
  await scrollToBottom()

  loading.value = true
  abortController = new AbortController()

  await streamQuery(
    question,
    null,
    {
      onAnalysis: ({ analysis, sources }) => {
        assistantMsg.analysis = analysis
        assistantMsg.sources = sources
        scrollToBottom()
      },
      onChunk: ({ content }) => {
        assistantMsg.content += content
        scrollToBottom()
      },
      onDone: ({ elapsed }) => {
        assistantMsg.streaming = false
        if (elapsed != null) assistantMsg.elapsed = elapsed
        loading.value = false
        abortController = null
      },
      onError: ({ message }) => {
        assistantMsg.streaming = false
        assistantMsg.error = true
        assistantMsg.content = assistantMsg.content
          ? `${assistantMsg.content}\n\n⚠️ ${message}`
          : `⚠️ ${message}`
        loading.value = false
        abortController = null
      },
    },
    abortController.signal,
  )

  // 流提前结束但 loading 未清（如 abort）
  if (loading.value) {
    assistantMsg.streaming = false
    loading.value = false
    abortController = null
  }
}

function stop() {
  if (abortController) {
    abortController.abort()
    abortController = null
  }
  loading.value = false
  const last = messages.value[messages.value.length - 1]
  if (last && last.role === 'assistant' && last.streaming) {
    last.streaming = false
    if (!last.content) last.content = '（已停止生成）'
  }
}

onMounted(() => {
  systemStore.startPolling()
})
</script>

<template>
  <div class="chat-view">
    <!-- 消息列表 -->
    <div ref="scrollRef" class="messages">
      <div v-if="!messages.length" class="empty">
        <div class="empty-icon">🍳</div>
        <h2>尝尝咸淡 · RAG 烹饪助手</h2>
        <p>基于图 RAG 的智能烹饪问答，问我任何关于菜谱、食材搭配、烹饪方法的问题。</p>
        <div class="suggestions">
          <el-button round @click="send('红烧肉怎么做？')">红烧肉怎么做？</el-button>
          <el-button round @click="send('鸡肉配什么蔬菜好？')">鸡肉配什么蔬菜好？</el-button>
          <el-button round @click="send('川菜有哪些特色菜？')">川菜有哪些特色菜？</el-button>
        </div>
      </div>

      <ChatMessage v-for="m in messages" :key="m.id" :message="m" />
    </div>

    <!-- 输入区 -->
    <MessageInput :loading="loading" :disabled="!systemStore.ready" @send="send" @stop="stop" />

    <!-- 未就绪提示 -->
    <el-alert
      v-if="!systemStore.ready"
      :title="systemStore.message || '系统未就绪'"
      type="warning"
      :closable="false"
      show-icon
      class="status-alert"
    />
  </div>
</template>

<style scoped>
.chat-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  position: relative;
}
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px 16px;
}
.empty {
  text-align: center;
  color: var(--el-text-color-secondary);
  padding: 60px 20px;
}
.empty-icon {
  font-size: 56px;
  margin-bottom: 12px;
}
.empty h2 {
  margin: 0 0 8px;
  color: var(--el-text-color-primary);
}
.empty p {
  margin: 0 0 24px;
}
.suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: center;
}
.status-alert {
  position: absolute;
  top: 12px;
  left: 50%;
  transform: translateX(-50%);
  max-width: 90%;
}
</style>
