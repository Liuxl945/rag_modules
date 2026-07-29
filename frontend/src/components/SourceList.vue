<script setup lang="ts">
import { computed } from 'vue'
import type { SourceDoc } from '@/types'

const props = defineProps<{ sources: SourceDoc[] }>()

// 检索通道中文标签
const channelLabel: Record<string, string> = {
  dual_level: '图键值双层',
  vector: '向量',
  bm25: 'BM25',
}
function chLabel(ch: string): string {
  return channelLabel[ch] || ch
}

// 检索方法/来源类型中文标签
const methodLabel: Record<string, string> = {
  dual_level: '图键值双层',
  vector: '向量检索',
  vector_enhanced: '向量增强',
  bm25: 'BM25 关键词',
  graph_path: '图路径',
  knowledge_subgraph: '知识子图',
}
function searchLabel(src: SourceDoc): string {
  const m = src.search_method || src.search_type
  return methodLabel[m] || m || '未知'
}

// 是否有可展开的详情（chunk 元信息、通道得分或重排得分）
function hasDetail(src: SourceDoc): boolean {
  return !!(
    src.chunk_id ||
    src.chunk_index != null ||
    src.section_title ||
    src.node_id ||
    src.rrf_sources?.length ||
    src.bm25_score != null ||
    src.vector_score != null ||
    src.dual_score != null ||
    src.reranked ||
    src.path_length != null
  )
}

// chunk 定位文案
function chunkLoc(src: SourceDoc): string {
  if (src.chunk_index != null && src.total_chunks != null) {
    return `chunk ${src.chunk_index + 1}/${src.total_chunks}`
  }
  if (src.chunk_index != null) return `chunk ${src.chunk_index + 1}`
  return ''
}

// 通道得分列表（用于详情展开）
interface ChannelScore {
  name: string
  label: string
  hit: boolean
  rank?: number
  raw?: number
}
function channelScores(src: SourceDoc): ChannelScore[] {
  const channels = ['dual_level', 'vector', 'bm25']
  const ranks = src.rrf_ranks || {}
  const raws = src.rrf_raw_scores || {}
  return channels.map((ch) => ({
    name: ch,
    label: chLabel(ch),
    hit: (src.rrf_sources || []).includes(ch),
    rank: ranks[ch],
    raw: raws[ch],
  }))
}

function fmtScore(v: number | null | undefined, digits = 3): string {
  if (v == null || Number.isNaN(v)) return '—'
  return v.toFixed(digits)
}

// 折叠面板默认展开第一个有详情的来源，方便用户直观看到细节
const activeNames = computed<string[]>(() => {
  const idx = props.sources.findIndex(hasDetail)
  return idx >= 0 ? [String(idx)] : []
})
</script>

<template>
  <el-collapse v-if="sources.length" class="source-list">
    <el-collapse-item :title="`检索来源（${sources.length}）`" name="sources">
      <el-collapse v-model="activeNames" class="source-inner">
        <el-collapse-item
          v-for="(src, i) in sources"
          :key="i"
          :name="String(i)"
          :disabled="!hasDetail(src)"
        >
          <template #title>
            <div class="source-head">
              <span class="source-name">{{ src.recipe_name }}</span>
              <el-tag size="small" type="info">{{ searchLabel(src) }}</el-tag>
              <el-tag v-if="src.reranked" size="small" type="success" effect="plain">🔁 重排</el-tag>
              <el-tag v-if="chunkLoc(src)" size="small" type="info" effect="plain">{{ chunkLoc(src) }}</el-tag>
              <span class="source-score">得分 {{ fmtScore(src.final_score ?? src.score) }}</span>
            </div>
          </template>

          <div v-if="hasDetail(src)" class="source-detail">
            <!-- chunk / 文档定位 -->
            <div v-if="src.section_title || src.node_id || src.chunk_id" class="detail-row">
              <span class="detail-label">文档定位</span>
              <span v-if="src.section_title" class="detail-tag">章节：{{ src.section_title }}</span>
              <span v-if="src.node_id" class="detail-tag">节点：{{ src.node_id }}</span>
              <span v-if="src.chunk_id" class="detail-tag">{{ src.chunk_id }}</span>
            </div>

            <!-- 图路径元信息 -->
            <div v-if="src.path_length != null || src.node_count != null" class="detail-row">
              <span class="detail-label">图结构</span>
              <span v-if="src.path_length != null" class="detail-tag">{{ src.path_length }} 跳</span>
              <span v-if="src.node_count != null" class="detail-tag">{{ src.node_count }} 节点</span>
              <span v-if="src.relationship_count != null" class="detail-tag">{{ src.relationship_count }} 关系</span>
            </div>

            <!-- 各检索通道命中与得分对比 -->
            <div v-if="src.rrf_sources?.length || src.bm25_score != null || src.vector_score != null || src.dual_score != null" class="detail-row">
              <span class="detail-label">通道命中</span>
              <div class="channel-grid">
                <div
                  v-for="ch in channelScores(src)"
                  :key="ch.name"
                  class="channel-cell"
                  :class="{ hit: ch.hit }"
                >
                  <span class="ch-name">{{ ch.label }}</span>
                  <span class="ch-status">{{ ch.hit ? '✓ 命中' : '—' }}</span>
                  <span v-if="ch.hit && ch.rank != null" class="ch-rank">排名 #{{ ch.rank }}</span>
                  <span v-if="ch.raw != null" class="ch-raw">原始分 {{ fmtScore(ch.raw, 4) }}</span>
                </div>
              </div>
            </div>

            <!-- 重排得分（cross-encoder 精排分） -->
            <div v-if="src.reranked && src.rerank_score != null" class="detail-row">
              <span class="detail-label">重排得分</span>
              <div class="rerank-cell">
                <span class="rerank-value">{{ fmtScore(src.rerank_score, 4) }}</span>
                <span class="rerank-hint">cross-encoder 精排分（已作为最终分）</span>
              </div>
            </div>

            <!-- 内容预览 -->
            <div class="detail-row">
              <span class="detail-label">内容预览</span>
              <div class="source-preview">{{ src.content_preview }}</div>
            </div>
          </div>
        </el-collapse-item>
      </el-collapse>
    </el-collapse-item>
  </el-collapse>
</template>

<style scoped>
.source-list {
  margin-top: 8px;
}
.source-inner {
  border: none;
}
.source-inner :deep(.el-collapse-item__header) {
  height: auto;
  padding: 4px 0;
  line-height: 1.4;
}
.source-head {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.source-name {
  font-weight: 600;
  color: var(--el-text-color-primary);
}
.source-score {
  color: var(--el-text-color-secondary);
  font-size: 12px;
  margin-left: auto;
}
.source-detail {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 4px 0 6px;
}
.detail-row {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  gap: 6px;
  font-size: 12px;
}
.detail-label {
  flex-shrink: 0;
  color: var(--el-text-color-secondary);
  min-width: 56px;
}
.detail-tag {
  color: var(--el-text-color-regular);
}
.channel-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
  width: 100%;
}
.channel-cell {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 4px 8px;
  border-radius: 6px;
  background: var(--el-fill-color-light);
  border: 1px solid var(--el-border-color-lighter);
  font-size: 12px;
}
.channel-cell.hit {
  background: var(--el-color-success-light-9);
  border-color: var(--el-color-success-light-5);
}
.ch-name {
  font-weight: 600;
  color: var(--el-text-color-primary);
}
.ch-status {
  color: var(--el-text-color-secondary);
}
.channel-cell.hit .ch-status {
  color: var(--el-color-success);
}
.ch-rank,
.ch-raw {
  color: var(--el-text-color-secondary);
  font-size: 11px;
}
.rerank-cell {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 8px;
  border-radius: 6px;
  background: var(--el-color-success-light-9);
  border: 1px solid var(--el-color-success-light-5);
  font-size: 12px;
}
.rerank-value {
  font-weight: 600;
  color: var(--el-color-success);
}
.rerank-hint {
  color: var(--el-text-color-secondary);
  font-size: 11px;
}
.source-preview {
  width: 100%;
  color: var(--el-text-color-regular);
  font-size: 13px;
  line-height: 1.5;
  white-space: pre-wrap;
  background: var(--el-fill-color-light);
  padding: 6px 8px;
  border-radius: 6px;
  max-height: 160px;
  overflow-y: auto;
}
</style>
