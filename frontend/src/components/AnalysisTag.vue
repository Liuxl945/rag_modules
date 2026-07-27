<script setup lang="ts">
import { computed } from 'vue'
import type { AnalysisInfo } from '@/types'

type TagType = 'primary' | 'success' | 'warning' | 'info' | 'danger'

const props = defineProps<{ analysis: AnalysisInfo }>()

// 策略中文标签映射
const strategyLabel = computed(() => {
  const map: Record<string, string> = {
    hybrid_traditional: '传统混合检索',
    graph_rag: '图 RAG 检索',
    combined: '组合策略',
  }
  return map[props.analysis.recommended_strategy] || props.analysis.recommended_strategy
})

// 策略对应的 Element Plus tag 类型
const strategyType = computed<TagType>(() => {
  const map: Record<string, TagType> = {
    hybrid_traditional: 'primary',
    graph_rag: 'success',
    combined: 'warning',
  }
  return map[props.analysis.recommended_strategy] || 'info'
})

// 复杂度/关系密集度分级文案
function levelLabel(v: number): string {
  if (v < 0.4) return '低'
  if (v < 0.8) return '中'
  return '高'
}
</script>

<template>
  <div class="analysis-tag">
    <el-tag :type="strategyType" size="small" effect="dark">{{ strategyLabel }}</el-tag>
    <el-tag size="small" type="info">复杂度 {{ analysis.query_complexity.toFixed(2) }}（{{ levelLabel(analysis.query_complexity) }}）</el-tag>
    <el-tag size="small" type="info">关系密集度 {{ analysis.relationship_intensity.toFixed(2) }}（{{ levelLabel(analysis.relationship_intensity) }}）</el-tag>
    <el-tag size="small" type="info">置信度 {{ analysis.confidence.toFixed(2) }}</el-tag>
    <el-tooltip :content="analysis.reasoning" placement="bottom" v-if="analysis.reasoning">
      <el-tag size="small" type="info" class="reasoning-tag">决策理由</el-tag>
    </el-tooltip>
  </div>
</template>

<style scoped>
.analysis-tag {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}
.reasoning-tag {
  cursor: help;
}
</style>
