<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { listRecipeNames, getRecipeGraph } from '@/api'
import type { KnowledgeGraph, KnowledgeNode, RecipeName } from '@/types'

const visible = defineModel<boolean>({ required: true })

const title = '菜谱知识图谱'

// 菜谱下拉
const recipeOptions = ref<RecipeName[]>([])
const selectedRid = ref<string | null>(null)
const loadingOptions = ref(false)

// 图谱状态
const containerRef = ref<HTMLDivElement | null>(null)
const loadingGraph = ref(false)
const error = ref('')
const graph = ref<KnowledgeGraph | null>(null)

let network: any = null

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

// vis-network 选项
const OPTIONS: any = {
  nodes: { shape: 'dot', size: 18, borderWidth: 2, font: { size: 13, face: 'sans-serif' } },
  edges: {
    arrows: { to: { enabled: true, scaleFactor: 0.5 } },
    color: { color: '#c0c4cc', highlight: '#409eff', opacity: 0.7 },
    smooth: { type: 'continuous' },
  },
  groups: {
    Recipe: { color: { background: '#409eff', border: '#337ecc' }, shape: 'star', size: 28 },
    Ingredient: { color: { background: '#67c23a', border: '#529b2e' } },
    CookingStep: { color: { background: '#e6a23c', border: '#b88230' } },
    Category: { color: { background: '#909399', border: '#72767b' }, shape: 'box' },
  },
  physics: {
    stabilization: { iterations: 200 },
    barnesHut: { gravitationalConstant: -8000, springLength: 130, springConstant: 0.04 },
  },
  interaction: { hover: true, tooltipDelay: 120 },
}

/**
 * 构建 vis-network tooltip HTML 元素。
 * 返回 HTMLElement 而非字符串，避免 vis-network 将内容当作纯文本渲染（转义）。
 */
function buildTooltip(n: KnowledgeNode): HTMLElement {
  const typeLabel: Record<string, string> = {
    Recipe: '菜谱',
    Ingredient: '食材',
    CookingStep: '烹饪步骤',
    Category: '分类',
  }
  const wrap = document.createElement('div')
  wrap.style.cssText = 'max-width:280px;line-height:1.6'
  const titleEl = document.createElement('b')
  titleEl.textContent = n.label
  wrap.appendChild(titleEl)
  wrap.appendChild(document.createElement('br'))
  const typeEl = document.createElement('span')
  typeEl.style.cssText = 'color:#909399;font-size:11px'
  typeEl.textContent = typeLabel[n.type] || n.type
  wrap.appendChild(typeEl)
  if (n.properties) {
    for (const [k, v] of Object.entries(n.properties)) {
      if (v == null || v === '') continue
      wrap.appendChild(document.createElement('br'))
      const span = document.createElement('span')
      span.style.cssText = 'color:#606266'
      span.textContent = `${PROP_LABELS[k] || k}: ${v}`
      wrap.appendChild(span)
    }
  }
  return wrap
}

function destroyNetwork() {
  if (network) {
    network.destroy()
    network = null
  }
}

async function loadAndRender(rid: string) {
  loadingGraph.value = true
  error.value = ''
  graph.value = null
  try {
    const g = await getRecipeGraph(rid)
    graph.value = g

    // 保证 dialog 内容已渲染完毕、containerRef 可用
    if (!containerRef.value || !g.nodes.length) return
    await new Promise((r) => setTimeout(r, 50))
    if (!containerRef.value) return

    destroyNetwork()
    // @ts-ignore vis-network/standalone 按官方文档运行时用法
    const { Network } = await import('vis-network/standalone')
    if (!containerRef.value) return

    const nodes = g.nodes.map((n) => ({
      id: n.id,
      label: n.label.length > 12 ? n.label.slice(0, 12) + '…' : n.label,
      group: n.type,
      title: buildTooltip(n),
    }))
    const edges = g.edges.map((e) => ({ from: e.from, to: e.to, arrows: 'to', title: e.type }))
    network = new Network(containerRef.value, { nodes, edges }, OPTIONS)
  } catch (e: any) {
    error.value = e?.response?.data?.detail || '加载知识图谱失败'
  } finally {
    loadingGraph.value = false
  }
}

function onRecipeChange(rid: string) {
  if (rid) loadAndRender(rid)
}

async function onOpened() {
  if (recipeOptions.value.length) return
  loadingOptions.value = true
  try {
    recipeOptions.value = await listRecipeNames()
  } catch {
    ElMessage.error('加载菜谱列表失败')
  } finally {
    loadingOptions.value = false
  }
}

/** el-select 过滤：匹配菜名或分类 */
function filterMethod(query: string, item: RecipeName): boolean {
  return item.name.includes(query) || item.category.includes(query)
}

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
    <!-- 菜谱下拉选择 -->
    <div class="kg-toolbar">
      <div class="kg-select-wrap">
        <el-select
          v-model="selectedRid"
          placeholder="请选择一个菜谱"
          filterable
          clearable
          :loading="loadingOptions"
          :filter-method="filterMethod"
          style="min-width: 320px"
          @change="onRecipeChange"
        >
          <el-option
            v-for="r in recipeOptions"
            :key="r.id"
            :label="`${r.name}（${r.category || '未分类'}）`"
            :value="r.id"
          >
            <span class="opt-name">{{ r.name }}</span>
            <span class="opt-cat">{{ r.category || '未分类' }}</span>
          </el-option>
        </el-select>
      </div>
      <span v-if="graph" class="kg-note">共 {{ graph.counts.total }} 个节点</span>
    </div>

    <!-- 图例 -->
    <div class="kg-legend">
      <span v-for="l in legend" :key="l.label" class="kg-legend-item">
        <i class="dot" :style="{ background: l.color }"></i>{{ l.label }}
      </span>
    </div>

    <!-- 图谱画布 -->
    <div v-loading="loadingGraph" class="kg-canvas-wrap">
      <el-alert v-if="error" :title="error" type="error" :closable="false" show-icon class="kg-error" />
      <div v-else-if="!selectedRid" class="kg-empty">
        <el-empty description="请从上方下拉框选择一个菜谱" :image-size="72" />
      </div>
      <div v-else-if="graph && !graph.nodes.length" class="kg-empty">
        <el-empty description="该菜谱暂无图谱数据" :image-size="80" />
      </div>
      <div ref="containerRef" class="kg-canvas"></div>
    </div>
  </el-dialog>
</template>

<style scoped>
.kg-toolbar {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 8px;
}
.kg-select-wrap {
  flex: 1;
}
.kg-note {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  flex-shrink: 0;
}
.opt-name {
  font-weight: 500;
}
.opt-cat {
  margin-left: 8px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.kg-legend {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 12px;
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
