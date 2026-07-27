import { defineStore } from 'pinia'
import { ref } from 'vue'
import { getHealth, getStats } from '@/api'
import type { HealthResponse, SystemStats } from '@/types'

/**
 * 系统状态 store
 * - 轮询 /api/health 维护后端就绪状态
 * - 拉取 /api/stats 供统计面板展示
 */
export const useSystemStore = defineStore('system', () => {
  const status = ref<HealthResponse['status']>('initializing')
  const ready = ref(false)
  const message = ref<string | null>(null)
  const stats = ref<SystemStats | null>(null)

  let pollTimer: ReturnType<typeof setInterval> | null = null

  async function refreshHealth() {
    try {
      const h = await getHealth()
      status.value = h.status
      ready.value = h.ready
      message.value = h.message
    } catch {
      // 后端未启动时视为初始化中
      status.value = 'initializing'
      ready.value = false
      message.value = '无法连接后端服务，请确认后端已启动'
    }
  }

  async function refreshStats() {
    try {
      stats.value = await getStats()
    } catch {
      stats.value = null
    }
  }

  /** 开始轮询健康状态，就绪后自动停止 */
  function startPolling() {
    if (pollTimer) return
    refreshHealth()
    pollTimer = setInterval(async () => {
      await refreshHealth()
      if (ready.value && pollTimer) {
        clearInterval(pollTimer)
        pollTimer = null
      }
    }, 3000)
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer)
      pollTimer = null
    }
  }

  return {
    status,
    ready,
    message,
    stats,
    refreshHealth,
    refreshStats,
    startPolling,
    stopPolling,
  }
})
