<script setup lang="ts">
import { ref, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { listRecipeNames, getRecipeDocument } from '@/api'
import type { RecipeDocument as RecipeDocumentType, RecipeName } from '@/types'

const visible = defineModel<boolean>({ required: true })
const props = defineProps<{ initialRid?: string | null }>()

const title = '菜谱文档详情'

// 菜谱下拉
const recipeOptions = ref<RecipeName[]>([])
const selectedRid = ref<string | null>(null)
const loadingOptions = ref(false)

// 文档状态
const loadingDoc = ref(false)
const error = ref('')
const doc = ref<RecipeDocumentType | null>(null)

const META_LABELS: Record<string, string> = {
  node_id: '菜谱ID',
  recipe_name: '菜谱名称',
  node_type: '节点类型',
  category: '分类',
  cuisine_type: '菜系',
  difficulty: '难度',
  prep_time: '准备时间',
  cook_time: '烹饪时间',
  servings: '份量',
  ingredients_count: '食材数量',
  steps_count: '步骤数量',
  doc_type: '文档类型',
  content_length: '内容长度',
}

async function loadDocument(rid: string) {
  loadingDoc.value = true
  error.value = ''
  doc.value = null
  try {
    doc.value = await getRecipeDocument(rid)
  } catch (e: any) {
    error.value = e?.response?.data?.detail || '加载文档失败'
  } finally {
    loadingDoc.value = false
  }
}

function onRecipeChange(rid: string) {
  if (rid) loadDocument(rid)
}

async function onOpened() {
  // 已有 options（之前加载过），直接选中 initialRid
  if (recipeOptions.value.length) {
    if (props.initialRid && selectedRid.value !== props.initialRid) {
      selectedRid.value = props.initialRid
      loadDocument(props.initialRid)
    }
    return
  }
  loadingOptions.value = true
  try {
    recipeOptions.value = await listRecipeNames()
    // 外部指定了 initialRid 时自动选中
    if (props.initialRid) {
      selectedRid.value = props.initialRid
      loadDocument(props.initialRid)
    }
  } catch {
    ElMessage.error('加载菜谱列表失败')
  } finally {
    loadingOptions.value = false
  }
}

// 对话框已打开时 initialRid 变化（如从浏览页直接切换菜谱）
watch(
  () => props.initialRid,
  (rid) => {
    if (rid && visible.value && recipeOptions.value.length) {
      selectedRid.value = rid
      loadDocument(rid)
    }
  }
)
</script>

<template>
  <el-dialog
    v-model="visible"
    :title="title"
    width="70%"
    top="5vh"
    :close-on-click-modal="false"
    @opened="onOpened"
  >
    <!-- 菜谱下拉选择 -->
    <div class="rd-toolbar">
      <div class="rd-select-wrap">
        <el-select
          v-model="selectedRid"
          placeholder="请选择一个菜谱"
          filterable
          clearable
          :loading="loadingOptions"
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
    </div>

    <!-- 文档内容区 -->
    <div v-loading="loadingDoc" class="rd-content-wrap">
      <el-alert v-if="error" :title="error" type="error" :closable="false" show-icon class="rd-error" />
      <div v-else-if="!selectedRid" class="rd-empty">
        <el-empty description="请从上方下拉框选择一个菜谱" :image-size="72" />
      </div>
      <template v-else-if="doc">
        <!-- 元数据标签 -->
        <div class="rd-meta">
          <el-tag
            v-for="(v, k) in doc.metadata"
            :key="k"
            size="small"
            class="rd-meta-tag"
          >
            {{ META_LABELS[k] || k }}: {{ v }}
          </el-tag>
        </div>
        <!-- 文档正文（markdown 渲染为 pre 保留格式） -->
        <div class="rd-doc-body">
          <pre>{{ doc.content }}</pre>
        </div>
      </template>
    </div>
  </el-dialog>
</template>

<style scoped>
.rd-toolbar {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 12px;
}
.rd-select-wrap {
  flex: 1;
}
.opt-name {
  font-weight: 500;
}
.opt-cat {
  margin-left: 8px;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
.rd-content-wrap {
  position: relative;
  max-height: 70vh;
  overflow-y: auto;
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 6px;
  background: var(--el-bg-color-page);
}
.rd-error {
  margin: 12px;
}
.rd-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 200px;
}
.rd-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  padding: 12px;
  border-bottom: 1px solid var(--el-border-color-lighter);
  background: var(--el-bg-color);
}
.rd-meta-tag {
  max-width: 100%;
}
.rd-doc-body {
  padding: 16px;
}
.rd-doc-body pre {
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial,
    'Noto Sans SC', sans-serif;
  font-size: 14px;
  line-height: 1.8;
  color: var(--el-text-color-primary);
}
</style>
