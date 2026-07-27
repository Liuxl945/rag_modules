<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted } from 'vue'
import ChatMessage from '@/components/ChatMessage.vue'
import MessageInput from '@/components/MessageInput.vue'
import ConversationSidebar from '@/components/ConversationSidebar.vue'
import { useSystemStore } from '@/stores/system'
import { useConversationStore } from '@/stores/conversations'

const systemStore = useSystemStore()
const store = useConversationStore()

const scrollRef = ref<HTMLDivElement | null>(null)

async function scrollToBottom() {
  await nextTick()
  const el = scrollRef.value
  if (el) el.scrollTop = el.scrollHeight
}

// 滚动触发键：消息数 + 末条内容长度 + 流式标记，覆盖新增/逐 token/状态切换
const scrollKey = computed(() => {
  const m = store.messages
  const last = m[m.length - 1]
  return `${m.length}:${last?.content?.length ?? 0}:${last?.streaming ? 1 : 0}`
})
watch(scrollKey, scrollToBottom)

function send(question: string) {
  store.sendMessage(question)
}

function stop() {
  store.stop()
}

onMounted(() => {
  systemStore.startPolling()
  store.fetchConversations()
})
</script>

<template>
  <div class="chat-view">
    <!-- 侧边栏：历史会话 -->
    <ConversationSidebar />

    <!-- 聊天主区 -->
    <div class="chat-main">
      <!-- 消息列表 -->
      <div ref="scrollRef" class="messages">
        <div v-if="!store.messages.length" class="empty">
          <div class="empty-icon">🍳</div>
          <h2>尝尝咸淡 · RAG 烹饪助手</h2>
          <p>基于图 RAG 的智能烹饪问答，问我任何关于菜谱、食材搭配、烹饪方法的问题。</p>
          <div class="suggestions">
            <el-button round @click="send('红烧肉怎么做？')">红烧肉怎么做？</el-button>
            <el-button round @click="send('鸡肉配什么蔬菜好？')">鸡肉配什么蔬菜好？</el-button>
            <el-button round @click="send('川菜有哪些特色菜？')">川菜有哪些特色菜？</el-button>
          </div>
        </div>

        <ChatMessage v-for="m in store.messages" :key="m.id" :message="m" />
      </div>

      <!-- 输入区 -->
      <MessageInput
        :loading="store.loading"
        :disabled="!systemStore.ready"
        @send="send"
        @stop="stop"
      />

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
  </div>
</template>

<style scoped>
.chat-view {
  display: flex;
  height: 100%;
}
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
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
