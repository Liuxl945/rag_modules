<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getStats, rebuildKnowledgeBase } from '@/api'
import type { SystemStats } from '@/types'

const stats = ref<SystemStats | null>(null)
const loading = ref(false)
const rebuilding = ref(false)

async function refresh() {
  loading.value = true
  try {
    stats.value = await getStats()
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '获取统计失败')
  } finally {
    loading.value = false
  }
}

async function rebuild() {
  try {
    await ElMessageBox.confirm(
      '这将删除现有向量数据并从 Neo4j 重新构建知识库，过程可能较长，是否继续？',
      '重建知识库',
      { confirmButtonText: '确认重建', cancelButtonText: '取消', type: 'warning' },
    )
  } catch {
    return // 取消
  }

  rebuilding.value = true
  try {
    const res = await rebuildKnowledgeBase()
    if (res.success) {
      ElMessage.success(res.message)
      stats.value = res.stats
    } else {
      ElMessage.error(res.message)
    }
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '重建失败')
  } finally {
    rebuilding.value = false
  }
}

// 知识库计数卡片
const kbCounts = computed(() => {
  const kb = stats.value?.knowledge_base || {}
  return [
    { label: '菜谱', value: (kb.total_recipes as number) ?? 0 },
    { label: '食材', value: (kb.total_ingredients as number) ?? 0 },
    { label: '烹饪步骤', value: (kb.total_cooking_steps as number) ?? 0 },
    { label: '文档', value: (kb.total_documents as number) ?? 0 },
    { label: '文本块', value: (kb.total_chunks as number) ?? 0 },
  ]
})

const vectorCount = computed(() => (stats.value?.milvus?.row_count as number) ?? 0)

const route = computed(() => stats.value?.route_stats)

const routeItems = computed(() => {
  const r = route.value
  if (!r) return []
  return [
    { label: '传统混合检索', count: r.traditional_count ?? 0, ratio: r.traditional_ratio ?? 0, color: '#409eff' },
    { label: '图 RAG 检索', count: r.graph_rag_count ?? 0, ratio: r.graph_rag_ratio ?? 0, color: '#67c23a' },
    { label: '组合策略', count: r.combined_count ?? 0, ratio: r.combined_ratio ?? 0, color: '#e6a23c' },
  ]
})

onMounted(refresh)
</script>

<template>
  <div class="stats-view">
    <div class="stats-header">
      <h2>系统统计</h2>
      <div>
        <el-button :loading="loading" @click="refresh">刷新</el-button>
        <el-button type="danger" :loading="rebuilding" @click="rebuild">重建知识库</el-button>
      </div>
    </div>

    <el-alert
      v-if="!stats?.ready"
      title="系统未就绪，统计信息可能不可用"
      type="warning"
      :closable="false"
      show-icon
      style="margin-bottom: 16px"
    />

    <!-- 知识库统计 -->
    <el-card shadow="never" class="section">
      <template #header>知识库统计</template>
      <el-row :gutter="16" justify="center">
        <el-col v-for="item in kbCounts" :key="item.label" :span="4">
          <el-statistic :title="item.label" :value="item.value" />
        </el-col>
      </el-row>
      <el-divider />
      <div class="vector-line">向量索引记录：<b>{{ vectorCount }}</b> 条</div>
    </el-card>

    <!-- 路由统计 -->
    <el-card shadow="never" class="section">
      <template #header>
        路由统计<span v-if="route" class="total">（总查询 {{ route.total_queries }} 次）</span>
      </template>
      <div v-if="route && route.total_queries > 0">
        <div v-for="item in routeItems" :key="item.label" class="route-row">
          <div class="route-label">{{ item.label }}</div>
          <el-progress
            :percentage="Math.round(item.ratio * 100)"
            :color="item.color"
            :stroke-width="18"
            :format="() => `${item.count} 次`"
          />
        </div>
      </div>
      <el-empty v-else description="暂无查询记录" :image-size="80" />
    </el-card>
  </div>
</template>

<style scoped>
.stats-view {
  padding: 20px;
  max-width: 960px;
  margin: 0 auto;
}
.stats-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.stats-header h2 {
  margin: 0;
}
.section {
  margin-bottom: 16px;
}
.vector-line {
  color: var(--el-text-color-regular);
}
.total {
  color: var(--el-text-color-secondary);
  font-size: 13px;
  font-weight: normal;
}
.route-row {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 14px;
}
.route-label {
  width: 120px;
  flex-shrink: 0;
  color: var(--el-text-color-regular);
}
.route-row :deep(.el-progress) {
  flex: 1;
}
</style>
