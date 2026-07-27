<script setup lang="ts">
import { ref, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useConversationStore } from '@/stores/conversations'
import type { ConversationMeta } from '@/types'

const store = useConversationStore()

// 折叠状态（自管）：折叠后只留窄条 + 展开按钮
const collapsed = ref(false)

const list = computed(() => store.conversations)

/** 相对时间：今天 HH:MM / 昨天 / 今年 MM-DD / 跨年 YYYY-MM-DD */
function formatTime(ts?: number): string {
  if (!ts) return ''
  const d = new Date(ts * 1000)
  const now = new Date()
  const sameDay = d.toDateString() === now.toDateString()
  const yest = new Date(now)
  yest.setDate(now.getDate() - 1)
  const isYest = d.toDateString() === yest.toDateString()
  const hh = String(d.getHours()).padStart(2, '0')
  const mm = String(d.getMinutes()).padStart(2, '0')
  if (sameDay) return `${hh}:${mm}`
  if (isYest) return '昨天'
  if (d.getFullYear() === now.getFullYear()) {
    return `${d.getMonth() + 1}月${d.getDate()}日`
  }
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

function select(id: string) {
  if (store.loading && id === store.activeId) return
  store.selectConversation(id)
}

async function onRename(c: ConversationMeta, e: Event) {
  e.stopPropagation()
  try {
    const { value } = await ElMessageBox.prompt('请输入新的会话标题', '重命名会话', {
      inputValue: c.title,
      inputPlaceholder: '会话标题',
      inputValidator: (v) => (v && v.trim().length > 0) || '标题不能为空',
      confirmButtonText: '保存',
      cancelButtonText: '取消',
    })
    if (value && value.trim() && value.trim() !== c.title) {
      await store.renameConversation(c.id, value.trim())
      ElMessage.success('已重命名')
    }
  } catch {
    /* 用户取消 */
  }
}

async function onDelete(c: ConversationMeta, e: Event) {
  e.stopPropagation()
  try {
    await ElMessageBox.confirm(`确定删除会话「${c.title}」？删除后不可恢复。`, '删除会话', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      confirmButtonClass: 'el-button--danger',
    })
    await store.deleteConversation(c.id)
    ElMessage.success('已删除')
  } catch {
    /* 用户取消 */
  }
}
</script>

<template>
  <aside class="conv-sidebar" :class="{ collapsed }">
    <!-- 顶部：新对话 + 折叠 -->
    <div class="sidebar-top">
      <el-button
        v-if="!collapsed"
        type="primary"
        class="new-btn"
        @click="store.startNewConversation()"
      >
        <span class="plus">➕</span> 新对话
      </el-button>
      <el-button text class="collapse-btn" @click="collapsed = !collapsed">
        {{ collapsed ? '▸' : '◂' }}
      </el-button>
    </div>

    <!-- 列表（折叠时隐藏） -->
    <div v-if="!collapsed" class="sidebar-list">
      <div v-if="!list.length && store.loaded" class="empty">暂无历史对话</div>
      <div
        v-for="c in list"
        :key="c.id"
        class="conv-item"
        :class="{ active: c.id === store.activeId }"
        @click="select(c.id)"
      >
        <div class="conv-main">
          <div class="conv-title">{{ c.title }}</div>
          <div class="conv-sub">
            <span class="conv-time">{{ formatTime(c.updated_at) }}</span>
            <span v-if="c.last_message_preview" class="conv-preview">
              · {{ c.last_message_preview }}
            </span>
          </div>
        </div>
        <div class="conv-actions" @click.stop>
          <button class="icon-btn" title="重命名" @click="onRename(c, $event)">✏️</button>
          <button class="icon-btn" title="删除" @click="onDelete(c, $event)">🗑️</button>
        </div>
      </div>
    </div>

    <!-- 折叠态：竖排新对话按钮 -->
    <div v-else class="sidebar-collapsed-body">
      <button class="icon-btn big" title="新对话" @click="store.startNewConversation()">➕</button>
    </div>
  </aside>
</template>

<style scoped>
.conv-sidebar {
  flex-shrink: 0;
  width: 240px;
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--el-border-color-lighter);
  background: var(--el-bg-color-page);
  transition: width 0.18s ease;
  overflow: hidden;
}
.conv-sidebar.collapsed {
  width: 44px;
}
.sidebar-top {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 10px 10px 8px;
}
.new-btn {
  flex: 1;
  justify-content: flex-start;
}
.new-btn .plus {
  margin-right: 4px;
}
.collapse-btn {
  flex-shrink: 0;
  padding: 6px 8px;
  color: var(--el-text-color-secondary);
  font-size: 14px;
}
.sidebar-list {
  flex: 1;
  overflow-y: auto;
  padding: 4px 8px 12px;
}
.sidebar-list .empty {
  text-align: center;
  color: var(--el-text-color-secondary);
  font-size: 13px;
  padding: 24px 0;
}
.conv-item {
  display: flex;
  align-items: flex-start;
  gap: 4px;
  padding: 8px 10px;
  border-radius: 8px;
  cursor: pointer;
  position: relative;
  transition: background 0.15s;
}
.conv-item:hover {
  background: var(--el-fill-color-light);
}
.conv-item.active {
  background: var(--el-color-primary-light-9);
}
.conv-main {
  flex: 1;
  min-width: 0;
}
.conv-title {
  font-size: 13.5px;
  color: var(--el-text-color-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  line-height: 1.4;
}
.conv-item.active .conv-title {
  color: var(--el-color-primary);
  font-weight: 600;
}
.conv-sub {
  margin-top: 2px;
  font-size: 11.5px;
  color: var(--el-text-color-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.conv-preview {
  color: var(--el-text-color-placeholder);
}
.conv-actions {
  display: flex;
  gap: 2px;
  opacity: 0;
  transition: opacity 0.15s;
  flex-shrink: 0;
}
.conv-item:hover .conv-actions {
  opacity: 1;
}
.icon-btn {
  border: none;
  background: transparent;
  cursor: pointer;
  padding: 2px 4px;
  border-radius: 4px;
  font-size: 13px;
  line-height: 1;
}
.icon-btn:hover {
  background: var(--el-fill-color-dark);
}
.icon-btn.big {
  font-size: 18px;
  padding: 6px;
}
.sidebar-collapsed-body {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 8px 0;
  gap: 8px;
}
</style>
