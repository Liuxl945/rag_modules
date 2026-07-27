<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{
  loading: boolean // 流式生成中
  disabled: boolean // 系统未就绪
}>()

const emit = defineEmits<{
  send: [question: string]
  stop: []
}>()

const text = ref('')

function send() {
  const q = text.value.trim()
  if (!q || props.loading || props.disabled) return
  emit('send', q)
  text.value = ''
}

// Ctrl/Cmd + Enter 或 Enter 发送（Shift+Enter 换行）
function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    send()
  }
}
</script>

<template>
  <div class="message-input">
    <el-input
      v-model="text"
      type="textarea"
      :rows="2"
      :autosize="{ minRows: 2, maxRows: 6 }"
      :placeholder="disabled ? '系统未就绪，请稍候...' : '输入你的烹饪问题，Enter 发送，Shift+Enter 换行'"
      :disabled="disabled"
      resize="none"
      @keydown="onKeydown"
    />
    <el-button
      v-if="!loading"
      type="primary"
      :disabled="!text.trim() || disabled"
      @click="send"
    >
      发送
    </el-button>
    <el-button v-else type="danger" plain @click="emit('stop')">
      停止
    </el-button>
  </div>
</template>

<style scoped>
.message-input {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  padding: 12px 16px;
  border-top: 1px solid var(--el-border-color-lighter);
  background: var(--el-bg-color);
}
.message-input :deep(.el-textarea__inner) {
  font-size: 14px;
}
</style>
