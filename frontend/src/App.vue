<script setup lang="ts">
import { onMounted, computed } from 'vue'
import { useSystemStore } from '@/stores/system'

type TagType = 'primary' | 'success' | 'warning' | 'info' | 'danger'

const systemStore = useSystemStore()

const statusText = computed(() => {
  switch (systemStore.status) {
    case 'ready':
      return '系统就绪'
    case 'error':
      return '系统异常'
    default:
      return '初始化中'
  }
})
const statusType = computed<TagType>(() => {
  switch (systemStore.status) {
    case 'ready':
      return 'success'
    case 'error':
      return 'danger'
    default:
      return 'warning'
  }
})

onMounted(() => {
  systemStore.startPolling()
})
</script>

<template>
  <el-container class="app-container">
    <el-header class="app-header">
      <div class="brand">
        <span class="logo">🍳</span>
        <span class="title">尝尝咸淡 · RAG 烹饪助手</span>
      </div>
      <el-menu mode="horizontal" router :default-active="$route.path" class="nav-menu">
        <el-menu-item index="/">聊天</el-menu-item>
        <el-menu-item index="/browse">菜谱</el-menu-item>
        <el-menu-item index="/stats">统计</el-menu-item>
      </el-menu>
      <el-tag :type="statusType" effect="dark" class="status-tag">
        {{ statusText }}
      </el-tag>
    </el-header>
    <el-main class="app-main">
      <router-view v-slot="{ Component }">
        <component :is="Component" />
      </router-view>
    </el-main>
  </el-container>
</template>

<style scoped>
.app-container {
  height: 100vh;
}
.app-header {
  display: flex;
  align-items: center;
  gap: 24px;
  border-bottom: 1px solid var(--el-border-color-lighter);
  background: var(--el-bg-color);
}
.brand {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  font-size: 16px;
}
.logo {
  font-size: 22px;
}
.nav-menu {
  flex: 1;
  border-bottom: none;
}
.status-tag {
  flex-shrink: 0;
}
.app-main {
  padding: 0;
  height: calc(100vh - 60px);
  overflow: hidden;
}
</style>
