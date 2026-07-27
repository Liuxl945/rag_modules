<script setup lang="ts">
import { ref, computed, watch, onUnmounted, nextTick } from 'vue'
import { getKnowledgeGraph } from '@/api'
import type { KnowledgeGraph, KnowledgeNode } from '@/types'

type GraphType = 'recipes' | 'ingredients' | 'cooking_steps'

const props = defineProps<{ type: GraphType | null }>()
const visible = defineModel<boolean>({ required: true })

const TYPE_TITLES: Record<GraphType, string> = {
  recipes: '菜谱',
  ingredients: '食材',
  cooking_steps: '烹饪步骤',
}
const PROP_LABELS: Record<string, string> = {
  category: '分类',
  cuisineType: '菜系',
  difficulty: '难度',
  servings: '份量',
  description: '描述',
  methods: '方法',
  tools: '工具',
  timeEstimate: '耗时',
}

const legend = [
  { label: '菜谱', color: '#409eff' },
  { label: '食材', color: '#67c23a' },
  { label: '烹饪步骤', color: '#e6a23c' },
  { label: '分类', color: '#909399' },
]

const title = computed(() => (props.type ? `${TYPE_TITLES[props.type]}知识图谱` : '知识图谱'))

const containerRef = ref<HTMLDivElement | null>(null)
const loading = ref(false)
const error = ref('')
const graph = ref<KnowledgeGraph | null>(null)
// vis-network 实例（非响应式，无需触发视图更新）
let network: any = null

// vis-network 选项（类型极其严格且冗长，用 any 规避繁琐的类型断言；运行时按文档配置）
const OPTIONS: any = {
  nodes: { shape: 'dot', size: 16, borderWidth: 2, font: { size: 13, face: 'sans-serif' } },
  edges: {
    arrows: { to: { enabled: true, scaleFactor: 0.5 } },
    color: { color: '#c0c4cc', highlight: '#409eff', opacity: 0.7 },
    smooth: { type: 'continuous' },
  },
  groups: {
    Recipe: { color: { background: '#409eff', border: '#337ecc' } },
    Ingredient: { color: { background: '#67c23a', border: '#529b2e' } },
    CookingStep: { color: { background: '#e6a23c', border: '#b88230' } },
    Category: { color: { background: '#909399', border: '#72767b' }, shape: 'box' },
  },
  physics: {
    stabilization: { iterations: 200 },
    barnesHut: { gravitationalConstant: -8000, springLength: 120, springConstant: 0.04 },
  },
  interaction: { hover: true, tooltipDelay: 120 },
}

function escapeHtml(s: string): string {
  return s.replace(
    /[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c] as string),
  )
}

function buildTooltip(n: KnowledgeNode): string {
  const typeLabel =
    ({ Recipe: '菜谱', Ingredient: '食材', CookingStep: '烹饪步骤', Category: '分类' } as Record<
      string,
      string
    >)[n.type] || n.type

  // 返回纯文本（换行分隔），避免 HTML 字符串在 tooltip 中被转义显示
  const lines: string[] = []
  lines.push(`${escapeHtml(n.label)} (${escapeHtml(typeLabel)})`)
  if (n.properties) {
    for (const [k, v] of Object.entries(n.properties)) {
      if (v == null || v === '') continue
      lines.push(`${PROP_LABELS[k] || k}: ${escapeHtml(String(v))}`)
    }
  }
  return lines.join('\n')
}

async function renderNetwork(g: KnowledgeGraph) {
  if (!containerRef.value || !g.nodes.length) return
  destroyNetwork()
  // @ts-ignore vis-network/standalone 类型解析不稳定，按官方文档运行时用法
  const { Network } = await import('vis-network/standalone')
  if (!containerRef.value) return // 异步期间可能已关闭
  const nodes = g.nodes.map((n) => ({
    id: n.id,
    label: n.label.length > 12 ? n.label.slice(0, 12) + '…' : n.label,
    group: n.type,
    title: buildTooltip(n),
  }))
  const edges = g.edges.map((e) => ({ from: e.from, to: e.to, arrows: 'to', title: e.type }))
  network = new Network(containerRef.value, { nodes, edges }, OPTIONS)
}

function destroyNetwork() {
  if (network) {
    network.destroy()
    network = null
  }
}

async function loadAndRender(t: GraphType) {
  loading.value = true
  error.value = ''
  graph.value = null
  try {
    const g = await getKnowledgeGraph(t)
    graph.value = g
    await nextTick()
    await renderNetwork(g)
  } catch (e: any) {
    error.value = e?.response?.data?.detail || '加载知识图谱失败'
  } finally {
    loading.value = false
  }
}

// 弹窗打开后（动画结束，容器已在 DOM）首次渲染
function onOpened() {
  if (props.type) loadAndRender(props.type)
}

// 弹窗已打开时切换类型 -> 重新加载
watch(
  () => props.type,
  (t) => {
    if (visible.value && t) loadAndRender(t)
  },
)

onUnmounted(destroyNetwork)
</script>

<template>
  <el-dialog
    v-model="visible"
    :title="title"
    width="80%"
    top="5vh"
    :close-on-click-modal="false"
    @opened="onOpened"
    @closed="destroyNetwork"
  >
    <div class="kg-toolbar">
      <div class="kg-legend">
        <span v-for="l in legend" :key="l.label" class="kg-legend-item">
          <i class="dot" :style="{ background: l.color }"></i>{{ l.label }}
        </span>
      </div>
      <span v-if="graph" class="kg-note">
        仅展示部分节点 · 主节点 {{ graph.counts.primary }} / 共 {{ graph.counts.total }} 个
      </span>
    </div>

    <div v-loading="loading" class="kg-canvas-wrap">
      <el-alert v-if="error" :title="error" type="error" :closable="false" show-icon class="kg-error" />
      <div v-else-if="graph && !graph.nodes.length" class="kg-empty">
        <el-empty description="暂无图谱数据" :image-size="80" />
      </div>
      <div ref="containerRef" class="kg-canvas"></div>
    </div>
  </el-dialog>
</template>

<style scoped>
.kg-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  flex-wrap: wrap;
  gap: 8px;
}
.kg-legend {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
.kg-legend-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: var(--el-text-color-regular);
}
.kg-legend-item .dot {
  display: inline-block;
  width: 12px;
  height: 12px;
  border-radius: 50%;
}
.kg-note {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.kg-canvas-wrap {
  position: relative;
  height: 65vh;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 6px;
  overflow: hidden;
  background: var(--el-bg-color-page);
}
.kg-canvas {
  width: 100%;
  height: 100%;
}
.kg-error {
  margin: 12px;
}
.kg-empty {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
