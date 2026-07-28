<script setup lang="ts">
import { computed } from 'vue'
import type { RetrievalTrace, GraphPath } from '@/types'

const props = defineProps<{ trace: RetrievalTrace }>()

// 查询类型中文标签
const queryTypeLabel: Record<string, string> = {
  entity_relation: '实体关系（一跳）',
  multi_hop: '多跳推理',
  subgraph: '知识子图',
  path_finding: '路径查找',
  clustering: '聚类相似',
}
function qTypeLabel(t: string): string {
  return queryTypeLabel[t] || t
}

// 通道中文标签
const channelLabel: Record<string, string> = {
  dual_level: '图键值双层',
  vector: '向量',
  bm25: 'BM25',
}
function chLabel(ch: string): string {
  return channelLabel[ch] || ch
}

// 是否有任何可展示内容
const hasContent = computed(() => {
  const t = props.trace
  return !!t.graph_query_plan || (t.graph_paths?.length ?? 0) > 0 || !!t.channel_stats
})

// 是否展示分数对比块
const showChannels = computed(() => !!props.trace.channel_stats)

// 是否展示图推理块
const showGraph = computed(
  () => !!props.trace.graph_query_plan || (props.trace.graph_paths?.length ?? 0) > 0,
)

// 最大跳数（用于标题摘要）
const maxHops = computed(() => {
  const hops = (props.trace.graph_paths || [])
    .map((p) => p.path_length ?? 0)
    .filter((h) => h > 0)
  return hops.length ? Math.max(...hops) : 0
})

// 面板标题的一行摘要
const titleSummary = computed(() => {
  const parts: string[] = []
  if (showGraph.value) {
    const n = props.trace.graph_paths?.length ?? 0
    if (n > 0) {
      parts.push(`图推理 ${n} 条路径${maxHops.value ? ` · 最深 ${maxHops.value} 跳` : ''}`)
    } else if (props.trace.graph_query_plan) {
      parts.push(`图推理 · ${qTypeLabel(props.trace.graph_query_plan.query_type)}`)
    }
  }
  if (showChannels.value) {
    parts.push('三路融合检索')
  }
  return parts.length ? parts.join(' · ') : '检索过程'
})

// 通道贡献明细（用于分数对比）
const channelRows = computed(() => {
  const cs = props.trace.channel_stats
  if (!cs) return []
  return cs.channels.map((ch) => ({
    name: ch,
    label: chLabel(ch),
    candidates: cs.candidates[ch] ?? 0,
    contributed: cs.contributed[ch] ?? 0,
    final: cs.final || 1,
  }))
})

// 路径节点标签颜色（按 labels 着色）
const labelColor: Record<string, string> = {
  Recipe: 'primary',
  Ingredient: 'success',
  CookingStep: 'warning',
  Category: 'info',
}
function nodeType(path: GraphPath, idx: number): string {
  const labels = path.nodes[idx]?.labels
  if (labels && labels.length) return labels[0]
  return ''
}
function nodeTagType(path: GraphPath, idx: number): string {
  const t = nodeType(path, idx)
  return labelColor[t] || 'info'
}

function fmtScore(v: number | null | undefined, digits = 3): string {
  if (v == null || Number.isNaN(v)) return '-'
  return v.toFixed(digits)
}
function fmtPct(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '-'
  return `${(v * 100).toFixed(0)}%`
}
</script>

<template>
  <el-collapse v-if="hasContent" class="trace-panel">
    <el-collapse-item :title="titleSummary" name="trace">
      <!-- C: 各检索策略分数对比 / 三路召回贡献 -->
      <div v-if="showChannels" class="trace-section">
        <div class="section-title">三路召回贡献（RRF 融合）</div>
        <div class="channel-table">
          <div class="channel-row channel-header">
            <span>通道</span>
            <span>候选数</span>
            <span>入选数</span>
            <span>入选占比</span>
          </div>
          <div v-for="row in channelRows" :key="row.name" class="channel-row">
            <span class="ch-name">{{ row.label }}</span>
            <span>{{ row.candidates }}</span>
            <span class="ch-contrib">{{ row.contributed }}</span>
            <span>
              <el-progress
                :percentage="Math.round((row.contributed / row.final) * 100)"
                :stroke-width="10"
                :show-text="false"
              />
              <span class="pct-text">{{ fmtPct(row.contributed / row.final) }}</span>
            </span>
          </div>
        </div>
        <div class="hint">各通道原始分口径不同（BM25 无界 / 向量余弦 0–1 / 双层规则分），以入选数与排名为主、原始分为辅，详见下方来源详情。</div>
      </div>

      <!-- B: 图 RAG 推理路径 -->
      <div v-if="showGraph" class="trace-section">
        <!-- 查询规划 -->
        <div v-if="trace.graph_query_plan" class="plan-card">
          <div class="section-title">查询规划</div>
          <div class="plan-row">
            <el-tag size="small" type="warning">{{ qTypeLabel(trace.graph_query_plan.query_type) }}</el-tag>
            <el-tag size="small" type="info">最大 {{ trace.graph_query_plan.max_depth }} 跳</el-tag>
          </div>
          <div class="plan-row">
            <span class="plan-label">源实体</span>
            <el-tag v-for="e in trace.graph_query_plan.source_entities" :key="'s'+e" size="small">{{ e }}</el-tag>
            <span v-if="!trace.graph_query_plan.source_entities?.length" class="muted">无</span>
          </div>
          <div v-if="trace.graph_query_plan.target_entities?.length" class="plan-row">
            <span class="plan-label">目标实体</span>
            <el-tag v-for="e in trace.graph_query_plan.target_entities" :key="'t'+e" size="small" type="success">{{ e }}</el-tag>
          </div>
          <div v-if="trace.graph_query_plan.relation_types?.length" class="plan-row">
            <span class="plan-label">关系类型</span>
            <el-tag v-for="r in trace.graph_query_plan.relation_types" :key="'r'+r" size="small" type="info" effect="plain">{{ r }}</el-tag>
          </div>
        </div>

        <!-- 推理路径列表 -->
        <div v-if="trace.graph_paths?.length" class="section-title">推理路径（{{ trace.graph_paths.length }}）</div>
        <div v-for="(p, i) in trace.graph_paths" :key="i" class="path-card">
          <div class="path-head">
            <span class="path-name">{{ p.recipe_name }}</span>
            <el-tag v-if="p.type === 'knowledge_subgraph'" size="small" type="info" effect="plain">知识子图</el-tag>
            <el-tag v-else size="small" type="success" effect="plain">图路径</el-tag>
            <el-tag v-if="p.path_length != null" size="small">{{ p.path_length }} 跳</el-tag>
            <span v-if="p.relevance_score != null" class="path-score">相关性 {{ fmtScore(p.relevance_score) }}</span>
          </div>

          <!-- 节点链 + 关系箭头 -->
          <div class="path-chain">
            <template v-for="(node, j) in p.nodes" :key="j">
              <el-tag size="small" :type="nodeTagType(p, j)" class="node-chip">
                {{ node.name || '节点' }}
              </el-tag>
              <span v-if="j < p.relationships.length" class="rel-arrow">
                <span class="rel-type">{{ p.relationships[j].type || '相关' }}</span>
                <span class="arrow">→</span>
              </span>
            </template>
            <span v-if="!p.nodes.length" class="muted">（无节点信息）</span>
          </div>

          <!-- 子图补充指标 -->
          <div v-if="p.type === 'knowledge_subgraph'" class="path-meta">
            <span v-if="p.node_count != null">{{ p.node_count }} 节点</span>
            <span v-if="p.relationship_count != null">{{ p.relationship_count }} 关系</span>
            <span v-if="p.graph_density != null">密度 {{ fmtScore(p.graph_density) }}</span>
          </div>
          <div v-if="p.reasoning_chains?.length" class="path-meta">
            <span class="plan-label">推理链</span>
            <el-tag v-for="(c, k) in p.reasoning_chains" :key="k" size="small" type="info" effect="plain">{{ c }}</el-tag>
          </div>
        </div>
      </div>
    </el-collapse-item>
  </el-collapse>
</template>

<style scoped>
.trace-panel {
  margin-top: 8px;
}
.trace-section {
  margin-bottom: 10px;
}
.trace-section:last-child {
  margin-bottom: 0;
}
.section-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--el-text-color-secondary);
  margin-bottom: 6px;
}
.hint {
  margin-top: 6px;
  font-size: 11px;
  color: var(--el-text-color-placeholder);
  line-height: 1.5;
}

/* 通道贡献表 */
.channel-table {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 12px;
}
.channel-row {
  display: grid;
  grid-template-columns: 1.2fr 0.8fr 0.8fr 1.8fr;
  align-items: center;
  gap: 8px;
  padding: 4px 6px;
  border-radius: 6px;
  background: var(--el-fill-color-light);
}
.channel-header {
  font-weight: 600;
  color: var(--el-text-color-secondary);
  background: transparent;
}
.ch-name {
  font-weight: 600;
}
.ch-contrib {
  color: var(--el-color-success);
  font-weight: 600;
}
.channel-row :deep(.el-progress) {
  display: inline-block;
  width: calc(100% - 44px);
  vertical-align: middle;
}
.pct-text {
  margin-left: 6px;
  color: var(--el-text-color-secondary);
}

/* 查询规划卡片 */
.plan-card {
  padding: 8px 10px;
  border-radius: 8px;
  background: var(--el-fill-color-light);
  margin-bottom: 8px;
}
.plan-row {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 6px;
}
.plan-label {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.muted {
  color: var(--el-text-color-placeholder);
  font-size: 12px;
}

/* 推理路径卡片 */
.path-card {
  padding: 8px 10px;
  border-radius: 8px;
  background: var(--el-fill-color-light);
  margin-bottom: 6px;
}
.path-head {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 6px;
}
.path-name {
  font-weight: 600;
  color: var(--el-text-color-primary);
  font-size: 13px;
}
.path-score {
  margin-left: auto;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.path-chain {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px;
}
.node-chip {
  font-weight: 600;
}
.rel-arrow {
  display: inline-flex;
  align-items: center;
  gap: 2px;
  color: var(--el-text-color-secondary);
  font-size: 11px;
}
.rel-type {
  background: var(--el-fill-color-dark);
  padding: 1px 5px;
  border-radius: 4px;
}
.arrow {
  color: var(--el-color-primary);
  font-weight: 700;
}
.path-meta {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 6px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>
