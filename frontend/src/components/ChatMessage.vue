<script setup lang="ts">
import { computed } from 'vue'
import type { ChatMessage } from '@/types'
import { renderMarkdown } from '@/utils/markdown'
import AnalysisTag from './AnalysisTag.vue'
import SourceList from './SourceList.vue'
import RetrievalTracePanel from './RetrievalTracePanel.vue'

const props = defineProps<{ message: ChatMessage }>()

const isUser = computed(() => props.message.role === 'user')
const htmlContent = computed(() => renderMarkdown(props.message.content))

// 是否有检索过程轨迹可展示（图推理路径或三路召回统计）
const hasTrace = computed(
  () =>
    !!props.message.retrieval_trace &&
    (!!props.message.retrieval_trace.graph_query_plan ||
      !!props.message.retrieval_trace.graph_paths?.length ||
      !!props.message.retrieval_trace.channel_stats),
)
</script>

<template>
  <div class="chat-message" :class="isUser ? 'is-user' : 'is-assistant'">
    <div class="avatar">{{ isUser ? '我' : '🍳' }}</div>
    <div class="bubble-wrap">
      <div class="bubble" :class="{ error: message.error }">
        <div v-if="message.content" class="content markdown-body" v-html="htmlContent"></div>
        <div v-else-if="message.streaming && !message.error" class="typing">
          正在思考<span class="dots"><span>.</span><span>.</span><span>.</span></span>
        </div>
        <span v-if="message.streaming && message.content" class="cursor">▋</span>
      </div>

      <!-- 路由分析 + 检索过程 + 检索来源（仅助手消息） -->
      <template v-if="!isUser">
        <AnalysisTag v-if="message.analysis" :analysis="message.analysis" class="meta" />
        <RetrievalTracePanel v-if="hasTrace" :trace="message.retrieval_trace!" />
        <SourceList v-if="message.sources && message.sources.length" :sources="message.sources" />
        <div v-if="message.elapsed != null && !message.streaming" class="elapsed">
          耗时 {{ message.elapsed.toFixed(2) }} 秒
        </div>
      </template>
    </div>
  </div>
</template>

<style scoped>
.chat-message {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
  align-items: flex-start;
}
.chat-message.is-user {
  flex-direction: row-reverse;
}
.avatar {
  flex-shrink: 0;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: var(--el-fill-color-light);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
}
.is-user .avatar {
  background: var(--el-color-primary);
  color: #fff;
  font-size: 14px;
}
.bubble-wrap {
  max-width: 75%;
  display: flex;
  flex-direction: column;
}
.is-user .bubble-wrap {
  align-items: flex-end;
}
.bubble {
  padding: 10px 14px;
  border-radius: 10px;
  background: var(--el-bg-color-page);
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
  word-break: break-word;
}
.is-user .bubble {
  background: var(--el-color-primary);
  color: #fff;
}
.bubble.error {
  background: var(--el-color-danger-light-9);
  color: var(--el-color-danger);
}
.content {
  line-height: 1.7;
  font-size: 14px;
}
.cursor {
  display: inline-block;
  animation: blink 1s step-start infinite;
  color: var(--el-color-primary);
}
.typing .dots span {
  animation: blink 1.2s infinite;
}
.typing .dots span:nth-child(2) {
  animation-delay: 0.2s;
}
.typing .dots span:nth-child(3) {
  animation-delay: 0.4s;
}
@keyframes blink {
  50% {
    opacity: 0;
  }
}
.meta {
  margin-top: 8px;
}
.elapsed {
  margin-top: 6px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>
