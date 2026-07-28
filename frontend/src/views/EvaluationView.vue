<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  getEvaluationStatus,
  getEvaluationDataset,
  evaluateSingle,
  evaluateMessage,
  runEvaluation,
  listEvaluationResults,
  getEvaluationResult,
  deleteEvaluationResult,
  listConversations,
  getConversation,
} from '@/api'
import type {
  EvaluationDatasetItem,
  EvaluationRunResult,
  EvaluationSingleResponse,
  EvaluationRunSummary,
  ChatMessage,
  ConversationMeta,
} from '@/types'

// ---------------------------------------------------------------------------
// 指标元信息（与后端 METRIC_META 对齐）
// ---------------------------------------------------------------------------
const METRIC_LABEL: Record<string, string> = {
  faithfulness: '忠实度',
  response_relevancy: '答案相关性',
  context_recall: '上下文召回率',
  context_precision: '上下文精确率',
  answer_relevancy: '答案相关性',
}
const METRIC_DESC: Record<string, string> = {
  faithfulness: '答案是否可由检索上下文支持（无幻觉）',
  response_relevancy: '答案是否切题（生成反推问题与原问题相似度）',
  context_recall: '参考答案是否都能被检索上下文覆盖',
  context_precision: '相关检索项是否排在前面（MRR 风格）',
  answer_relevancy: '答案是否切题',
}
function metricLabel(k: string): string {
  return METRIC_LABEL[k] || k
}
function metricDesc(k: string): string {
  return METRIC_DESC[k] || k
}

function scoreColor(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '#909399'
  if (v >= 0.7) return '#67c23a'
  if (v >= 0.4) return '#e6a23c'
  return '#f56c6c'
}
function fmtScore(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '-'
  return v.toFixed(3)
}

// ---------------------------------------------------------------------------
// 依赖可用性
// ---------------------------------------------------------------------------
const ragasAvailable = ref(true)
const statusLoaded = ref(false)
async function loadStatus() {
  try {
    const s = await getEvaluationStatus()
    ragasAvailable.value = s.available
  } catch {
    ragasAvailable.value = false
  } finally {
    statusLoaded.value = true
  }
}

// ---------------------------------------------------------------------------
// 测试集评估
// ---------------------------------------------------------------------------
const activeTab = ref('dataset')
const dataset = ref<EvaluationDatasetItem[]>([])
const datasetLoading = ref(false)
const running = ref(false)
const runResult = ref<EvaluationRunResult | null>(null)
const progressText = ref('')

async function loadDataset() {
  datasetLoading.value = true
  try {
    dataset.value = await getEvaluationDataset()
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '加载测试集失败')
  } finally {
    datasetLoading.value = false
  }
}

async function runDatasetEval() {
  if (!ragasAvailable.value) {
    ElMessage.warning('RAGAS 未安装，请先在后端安装评估依赖')
    return
  }
  running.value = true
  progressText.value = `正在对 ${dataset.value.length} 条样本跑完整 RAG 管线 + RAGAS 评估，可能需要数分钟…`
  try {
    runResult.value = await runEvaluation()
    ElMessage.success(`评估完成（${runResult.value.count} 条，耗时 ${runResult.value.elapsed ?? '-'}s）`)
    loadHistory()
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '评估失败')
  } finally {
    running.value = false
    progressText.value = ''
  }
}

const aggregateItems = computed(() => {
  const r = runResult.value
  if (!r) return []
  return r.metrics.map((k) => ({
    key: k,
    label: metricLabel(k),
    value: r.aggregates[k] ?? null,
  }))
})

// ---------------------------------------------------------------------------
// 单条评估 - 手动输入
// ---------------------------------------------------------------------------
const singleForm = ref({
  question: '',
  answer: '',
  contextsText: '',  // 多行文本，每行一个 context
  groundTruth: '',
})
const singleResult = ref<EvaluationSingleResponse | null>(null)
const singleLoading = ref(false)

async function runSingle() {
  if (!ragasAvailable.value) {
    ElMessage.warning('RAGAS 未安装，请先在后端安装评估依赖')
    return
  }
  if (!singleForm.value.question.trim() || !singleForm.value.answer.trim()) {
    ElMessage.warning('请填写问题和回答')
    return
  }
  const contexts = singleForm.value.contextsText
    .split('\n')
    .map((s) => s.trim())
    .filter(Boolean)
  singleLoading.value = true
  try {
    singleResult.value = await evaluateSingle({
      question: singleForm.value.question.trim(),
      answer: singleForm.value.answer.trim(),
      contexts,
      ground_truth: singleForm.value.groundTruth.trim() || undefined,
    })
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '评估失败')
  } finally {
    singleLoading.value = false
  }
}

const singleScoreItems = computed(() => {
  const r = singleResult.value
  if (!r) return []
  return r.metrics.map((k) => ({ key: k, label: metricLabel(k), value: r.scores[k] ?? null }))
})

// ---------------------------------------------------------------------------
// 单条评估 - 从会话选择
// ---------------------------------------------------------------------------
const conversations = ref<ConversationMeta[]>([])
const selConvId = ref<string | null>(null)
const convMessages = ref<ChatMessage[]>([])
const selMsgId = ref<string | null>(null)
const msgLoading = ref(false)
const msgResult = ref<(EvaluationSingleResponse & { question?: string }) | null>(null)

async function loadConvList() {
  try {
    conversations.value = await listConversations()
  } catch {
    conversations.value = []
  }
}

async function onConvChange(id: string) {
  selMsgId.value = null
  msgResult.value = null
  convMessages.value = []
  if (!id) return
  try {
    const conv = await getConversation(id)
    // 仅展示有内容的助手消息
    convMessages.value = (conv.messages || []).filter(
      (m) => m.role === 'assistant' && m.content,
    )
  } catch {
    convMessages.value = []
  }
}

async function runMessageEval() {
  if (!ragasAvailable.value) {
    ElMessage.warning('RAGAS 未安装，请先在后端安装评估依赖')
    return
  }
  if (!selConvId.value || !selMsgId.value) {
    ElMessage.warning('请选择会话和回答消息')
    return
  }
  msgLoading.value = true
  msgResult.value = null
  try {
    msgResult.value = await evaluateMessage(selConvId.value, selMsgId.value)
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '评估失败')
  } finally {
    msgLoading.value = false
  }
}

const msgScoreItems = computed(() => {
  const r = msgResult.value
  if (!r) return []
  return r.metrics.map((k) => ({ key: k, label: metricLabel(k), value: r.scores[k] ?? null }))
})

// ---------------------------------------------------------------------------
// 历史评估
// ---------------------------------------------------------------------------
const history = ref<EvaluationRunSummary[]>([])
const historyLoading = ref(false)
const detailVisible = ref(false)
const detail = ref<EvaluationRunResult | null>(null)

async function loadHistory() {
  historyLoading.value = true
  try {
    history.value = await listEvaluationResults()
  } catch {
    history.value = []
  } finally {
    historyLoading.value = false
  }
}

async function viewDetail(id: string) {
  try {
    detail.value = await getEvaluationResult(id)
    detailVisible.value = true
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '加载详情失败')
  }
}

async function removeRun(id: string) {
  try {
    await ElMessageBox.confirm('删除该评估记录？', '确认', {
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      type: 'warning',
    })
  } catch {
    return
  }
  try {
    await deleteEvaluationResult(id)
    ElMessage.success('已删除')
    loadHistory()
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '删除失败')
  }
}

function fmtTime(ts: number | null | undefined): string {
  if (!ts) return '-'
  const d = new Date(ts * 1000)
  return d.toLocaleString()
}

const detailAggItems = computed(() => {
  const r = detail.value
  if (!r) return []
  return r.metrics.map((k) => ({ key: k, label: metricLabel(k), value: r.aggregates[k] ?? null }))
})

onMounted(() => {
  loadStatus()
  loadDataset()
  loadConvList()
  loadHistory()
})
</script>

<template>
  <div class="eval-view">
    <div class="eval-header">
      <h2>RAGAS 评估</h2>
      <div class="header-actions">
        <el-button @click="loadStatus">检查依赖</el-button>
        <el-button @click="loadDataset">刷新测试集</el-button>
      </div>
    </div>

    <!-- 依赖未就绪横幅 -->
    <el-alert
      v-if="statusLoaded && !ragasAvailable"
      title="RAGAS 评估依赖未安装"
      type="warning"
      :closable="false"
      show-icon
      style="margin-bottom: 16px"
    >
      <template #default>
        评估接口不可用。请在后端执行：<code>pip install ragas langchain-openai</code>
        （主应用与聊天功能不受影响）。
      </template>
    </el-alert>

    <el-tabs v-model="activeTab" class="eval-tabs">
      <!-- 测试集评估 -->
      <el-tab-pane label="测试集评估" name="dataset">
        <el-card shadow="never" class="section">
          <template #header>
            <div class="card-header">
              <span>内置烹饪测试集</span>
              <div>
                <el-tag type="info" size="small">{{ dataset.length }} 条</el-tag>
                <el-button
                  type="primary"
                  :loading="running"
                  @click="runDatasetEval"
                  style="margin-left: 12px"
                >
                  运行评估
                </el-button>
              </div>
            </div>
          </template>

          <el-alert
            v-if="progressText"
            :title="progressText"
            type="info"
            :closable="false"
            show-icon
            :description="running ? '每条样本都会执行 路由→检索→生成→RAGAS，请耐心等待' : ''"
            style="margin-bottom: 12px"
          />

          <el-table
            v-loading="datasetLoading"
            :data="dataset"
            border
            size="small"
            style="width: 100%"
          >
            <el-table-column type="index" label="#" width="50" />
            <el-table-column prop="category" label="分类" width="100" />
            <el-table-column prop="question" label="问题" min-width="200" show-overflow-tooltip />
            <el-table-column prop="ground_truth" label="参考答案" min-width="260" show-overflow-tooltip />
          </el-table>
        </el-card>

        <!-- 评估结果 -->
        <el-card v-if="runResult" shadow="never" class="section">
          <template #header>
            <div class="card-header">
              <span>评估结果</span>
              <span class="muted">
                {{ runResult.count }} 条样本 · 耗时 {{ runResult.elapsed ?? '-' }}s
                <span v-if="runResult.skipped"> · 跳过 {{ runResult.skipped }} 条</span>
              </span>
            </div>
          </template>

          <!-- 聚合指标卡片 -->
          <el-row :gutter="16" style="margin-bottom: 16px">
            <el-col v-for="item in aggregateItems" :key="item.key" :span="6">
              <div class="agg-card">
                <div class="agg-label" :title="metricDesc(item.key)">{{ item.label }}</div>
                <div class="agg-value" :style="{ color: scoreColor(item.value) }">
                  {{ fmtScore(item.value) }}
                </div>
                <el-progress
                  :percentage="Math.round((item.value ?? 0) * 100)"
                  :color="scoreColor(item.value)"
                  :show-text="false"
                  :stroke-width="6"
                />
              </div>
            </el-col>
          </el-row>

          <!-- 每样本明细 -->
          <el-table :data="runResult.results" border size="small" style="width: 100%">
            <el-table-column type="index" label="#" width="50" />
            <el-table-column prop="question" label="问题" min-width="200" show-overflow-tooltip />
            <el-table-column
              v-for="m in runResult.metrics"
              :key="m"
              :label="metricLabel(m)"
              width="130"
            >
              <template #default="{ row }">
                <span :style="{ color: scoreColor(row[m]) }">{{ fmtScore(row[m]) }}</span>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-tab-pane>

      <!-- 单条评估 -->
      <el-tab-pane label="单条评估" name="single">
        <el-card shadow="never" class="section">
          <template #header>从会话选择</template>
          <div class="single-row">
            <el-select
              v-model="selConvId"
              placeholder="选择会话"
              filterable
              clearable
              style="width: 320px"
              @change="onConvChange"
            >
              <el-option
                v-for="c in conversations"
                :key="c.id"
                :label="c.title"
                :value="c.id"
              />
            </el-select>
            <el-select
              v-model="selMsgId"
              placeholder="选择助手回答"
              filterable
              clearable
              style="width: 360px"
            >
              <el-option
                v-for="m in convMessages"
                :key="m.id"
                :label="(m.content || '').slice(0, 40) + ((m.content || '').length > 40 ? '…' : '')"
                :value="m.id"
              />
            </el-select>
            <el-button type="primary" :loading="msgLoading" @click="runMessageEval">
              评估此回答
            </el-button>
          </div>
          <div class="muted hint">
            评估会重新检索取完整上下文（存储的来源仅含 300 字预览），并对已生成的回答打分。
            无参考答案，仅计算忠实度与答案相关性。
          </div>

          <div v-if="msgLoading" class="muted" style="margin-top: 12px">评估中，请稍候…</div>
          <div v-if="msgResult" class="score-grid">
            <div v-for="item in msgScoreItems" :key="item.key" class="agg-card small">
              <div class="agg-label" :title="metricDesc(item.key)">{{ item.label }}</div>
              <div class="agg-value" :style="{ color: scoreColor(item.value) }">
                {{ fmtScore(item.value) }}
              </div>
            </div>
          </div>
        </el-card>

        <el-card shadow="never" class="section">
          <template #header>手动输入</template>
          <el-form label-position="top">
            <el-form-item label="问题">
              <el-input v-model="singleForm.question" placeholder="用户的问题" />
            </el-form-item>
            <el-form-item label="回答">
              <el-input
                v-model="singleForm.answer"
                type="textarea"
                :rows="3"
                placeholder="待评估的回答"
              />
            </el-form-item>
            <el-form-item label="检索上下文（每行一个）">
              <el-input
                v-model="singleForm.contextsText"
                type="textarea"
                :rows="4"
                placeholder="检索到的上下文片段，每行一个"
              />
            </el-form-item>
            <el-form-item label="参考答案（可选，提供则跑完整 4 指标）">
              <el-input
                v-model="singleForm.groundTruth"
                type="textarea"
                :rows="2"
                placeholder="标准答案（可选）"
              />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="singleLoading" @click="runSingle">
                评估
              </el-button>
            </el-form-item>
          </el-form>

          <div v-if="singleResult" class="score-grid">
            <div v-for="item in singleScoreItems" :key="item.key" class="agg-card small">
              <div class="agg-label" :title="metricDesc(item.key)">{{ item.label }}</div>
              <div class="agg-value" :style="{ color: scoreColor(item.value) }">
                {{ fmtScore(item.value) }}
              </div>
            </div>
          </div>
        </el-card>
      </el-tab-pane>

      <!-- 历史评估 -->
      <el-tab-pane label="历史评估" name="history">
        <el-card shadow="never" class="section">
          <template #header>
            <div class="card-header">
              <span>历史评估记录</span>
              <el-button size="small" @click="loadHistory">刷新</el-button>
            </div>
          </template>
          <el-table v-loading="historyLoading" :data="history" border size="small" style="width: 100%">
            <el-table-column prop="id" label="ID" width="90" />
            <el-table-column label="时间" width="170">
              <template #default="{ row }">{{ fmtTime(row.created_at) }}</template>
            </el-table-column>
            <el-table-column prop="kind" label="类型" width="100" />
            <el-table-column prop="count" label="样本数" width="80" />
            <el-table-column label="耗时(s)" width="90">
              <template #default="{ row }">{{ row.elapsed ?? '-' }}</template>
            </el-table-column>
            <el-table-column label="聚合得分" min-width="240">
              <template #default="{ row }">
                <el-tag
                  v-for="m in row.metrics"
                  :key="m"
                  size="small"
                  effect="plain"
                  style="margin-right: 6px"
                >
                  {{ metricLabel(m) }}: {{ fmtScore(row.aggregates[m]) }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column label="操作" width="140">
              <template #default="{ row }">
                <el-button size="small" link @click="viewDetail(row.id)">详情</el-button>
                <el-button size="small" link type="danger" @click="removeRun(row.id)">删除</el-button>
              </template>
            </el-table-column>
          </el-table>
          <el-empty v-if="!historyLoading && !history.length" description="暂无评估记录" :image-size="80" />
        </el-card>
      </el-tab-pane>
    </el-tabs>

    <!-- 详情弹窗 -->
    <el-dialog v-model="detailVisible" title="评估详情" width="80%">
      <template v-if="detail">
        <div class="muted" style="margin-bottom: 12px">
          {{ detail.count }} 条样本 · 耗时 {{ detail.elapsed ?? '-' }}s · ID {{ detail.run_id }}
        </div>
        <el-row :gutter="16" style="margin-bottom: 16px">
          <el-col v-for="item in detailAggItems" :key="item.key" :span="6">
            <div class="agg-card">
              <div class="agg-label" :title="metricDesc(item.key)">{{ item.label }}</div>
              <div class="agg-value" :style="{ color: scoreColor(item.value) }">
                {{ fmtScore(item.value) }}
              </div>
            </div>
          </el-col>
        </el-row>
        <el-table :data="detail.results" border size="small" style="width: 100%">
          <el-table-column type="index" label="#" width="50" />
          <el-table-column prop="question" label="问题" min-width="200" show-overflow-tooltip />
          <el-table-column
            v-for="m in detail.metrics"
            :key="m"
            :label="metricLabel(m)"
            width="130"
          >
            <template #default="{ row }">
              <span :style="{ color: scoreColor(row[m]) }">{{ fmtScore(row[m]) }}</span>
            </template>
          </el-table-column>
        </el-table>
      </template>
    </el-dialog>
  </div>
</template>

<style scoped>
.eval-view {
  padding: 20px;
  max-width: 1100px;
  margin: 0 auto;
}
.eval-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.eval-header h2 {
  margin: 0;
}
.header-actions {
  display: flex;
  gap: 8px;
}
.eval-tabs {
  margin-top: 4px;
}
.section {
  margin-bottom: 16px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.muted {
  color: var(--el-text-color-secondary);
  font-size: 13px;
}
.hint {
  margin-top: 8px;
  line-height: 1.5;
}
.agg-card {
  padding: 12px;
  border-radius: 8px;
  background: var(--el-fill-color-light);
  border: 1px solid var(--el-border-color-lighter);
  text-align: center;
}
.agg-card.small {
  padding: 14px 12px;
}
.agg-label {
  font-size: 13px;
  color: var(--el-text-color-regular);
  margin-bottom: 6px;
  cursor: help;
}
.agg-value {
  font-size: 24px;
  font-weight: 700;
  margin-bottom: 8px;
}
.agg-card.small .agg-value {
  font-size: 22px;
}
.single-row {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}
.score-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 12px;
  margin-top: 12px;
}
code {
  background: var(--el-fill-color-dark);
  padding: 1px 6px;
  border-radius: 4px;
  font-family: 'SFMono-Regular', Consolas, monospace;
}
</style>
