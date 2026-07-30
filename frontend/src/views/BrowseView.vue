<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Upload } from '@element-plus/icons-vue'
import { listRecipesFull } from '@/api'
import type { RecipeListItem, UploadRecipeResponse } from '@/types'
import KnowledgeGraphDialog from '@/components/KnowledgeGraphDialog.vue'
import RecipeDocumentDialog from '@/components/RecipeDocumentDialog.vue'
import RecipeUploadDialog from '@/components/RecipeUploadDialog.vue'

const router = useRouter()

const recipes = ref<RecipeListItem[]>([])
const loading = ref(false)

// 搜索和筛选
const searchQuery = ref('')
const filterCategory = ref('')
const filterDifficulty = ref<number | ''>('')
const filterSource = ref('')

// 弹窗
const graphVisible = ref(false)
const docVisible = ref(false)
const uploadVisible = ref(false)
const selectedRid = ref<string | null>(null)

async function fetchRecipes() {
  loading.value = true
  try {
    recipes.value = await listRecipesFull()
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '加载菜谱列表失败')
  } finally {
    loading.value = false
  }
}

// 筛选选项
const categories = computed(() => {
  const set = new Set<string>()
  recipes.value.forEach((r) => r.category && set.add(r.category))
  return Array.from(set).sort()
})

const sources = computed(() => {
  const set = new Set<string>()
  recipes.value.forEach((r) => r.source && set.add(r.source))
  return Array.from(set).sort()
})

const sourceLabel: Record<string, string> = {
  neo4j: '内置菜谱',
  markdown_upload: '用户上传',
}

// 筛选后的菜谱列表
const filteredRecipes = computed(() => {
  let list = recipes.value
  const q = searchQuery.value.trim().toLowerCase()
  if (q) {
    list = list.filter((r) => r.name.toLowerCase().includes(q) || r.description.toLowerCase().includes(q))
  }
  if (filterCategory.value) {
    list = list.filter((r) => r.category === filterCategory.value)
  }
  if (filterDifficulty.value !== '') {
    list = list.filter((r) => r.difficulty === filterDifficulty.value)
  }
  if (filterSource.value) {
    list = list.filter((r) => r.source === filterSource.value)
  }
  return list
})

function openGraph(rid: string) {
  selectedRid.value = rid
  graphVisible.value = true
}

function openDocument(rid: string) {
  selectedRid.value = rid
  docVisible.value = true
}

function askAbout(name: string) {
  router.push({ path: '/', query: { ask: `${name}怎么做？` } })
}

function difficultyStars(n: number): string {
  if (!n) return '-'
  return '★'.repeat(Math.min(n, 5)) + '☆'.repeat(Math.max(0, 5 - n))
}

function resetFilters() {
  searchQuery.value = ''
  filterCategory.value = ''
  filterDifficulty.value = ''
  filterSource.value = ''
}

function onUploaded(_res: UploadRecipeResponse) {
  // 上传成功后刷新菜谱列表
  fetchRecipes()
}

onMounted(fetchRecipes)
</script>

<template>
  <div class="browse-view">
    <div class="browse-header">
      <h2>菜谱浏览</h2>
      <span class="count">共 {{ recipes.length }} 道菜谱</span>
      <div class="header-actions">
        <el-button type="primary" :icon="Upload" @click="uploadVisible = true">
          上传菜谱
        </el-button>
      </div>
    </div>

    <!-- 搜索和筛选栏 -->
    <el-card shadow="never" class="filter-card">
      <el-row :gutter="16" align="middle">
        <el-col :span="7">
          <el-input
            v-model="searchQuery"
            placeholder="搜索菜谱名称或描述..."
            clearable
          >
            <template #prefix>🔍</template>
          </el-input>
        </el-col>
        <el-col :span="4">
          <el-select v-model="filterCategory" placeholder="全部分类" clearable style="width: 100%">
            <el-option v-for="c in categories" :key="c" :label="c" :value="c" />
          </el-select>
        </el-col>
        <el-col :span="4">
          <el-select v-model="filterDifficulty" placeholder="全部难度" clearable style="width: 100%">
            <el-option v-for="d in [1,2,3,4,5]" :key="d" :label="difficultyStars(d)" :value="d" />
          </el-select>
        </el-col>
        <el-col :span="4">
          <el-select v-model="filterSource" placeholder="全部来源" clearable style="width: 100%">
            <el-option v-for="s in sources" :key="s" :label="sourceLabel[s] || s" :value="s" />
          </el-select>
        </el-col>
        <el-col :span="4">
          <el-button @click="resetFilters">重置</el-button>
          <el-button type="primary" :loading="loading" @click="fetchRecipes">刷新</el-button>
        </el-col>
      </el-row>
      <div v-if="filteredRecipes.length !== recipes.length" class="filter-result">
        筛选结果：{{ filteredRecipes.length }} 道
      </div>
    </el-card>

    <!-- 菜谱卡片网格 -->
    <div v-loading="loading" class="recipe-grid">
      <el-empty
        v-if="!loading && !filteredRecipes.length"
        :description="recipes.length ? '没有匹配的菜谱' : '暂无菜谱数据'"
        :image-size="100"
      />
      <el-card
        v-for="r in filteredRecipes"
        :key="r.id"
        shadow="hover"
        class="recipe-card"
      >
        <div class="card-top">
          <h3 class="card-title" :title="r.name">{{ r.name }}</h3>
          <el-tag v-if="r.source === 'markdown_upload'" size="small" type="success" effect="plain">上传</el-tag>
        </div>

        <div class="card-meta">
          <span class="meta-item" v-if="r.category">
            <el-tag size="small" type="info" effect="plain">{{ r.category }}</el-tag>
          </span>
          <span class="meta-item" v-if="r.difficulty">
            <span class="stars" :title="`难度: ${r.difficulty}星`">{{ difficultyStars(r.difficulty) }}</span>
          </span>
        </div>

        <p v-if="r.description" class="card-desc">{{ r.description }}</p>

        <div class="card-stats">
          <span title="食材数量">🥬 {{ r.ingredients_count || 0 }} 种食材</span>
          <span title="步骤数量">📝 {{ r.steps_count || 0 }} 个步骤</span>
        </div>

        <div class="card-actions">
          <el-button size="small" @click="askAbout(r.name)">💬 问问它</el-button>
          <el-button size="small" @click="openGraph(r.id)">🔗 图谱</el-button>
          <el-button size="small" type="primary" @click="openDocument(r.id)">📄 文档</el-button>
        </div>
      </el-card>
    </div>

    <!-- 知识图谱弹窗（带外部触发选中） -->
    <KnowledgeGraphDialog v-model="graphVisible" :initial-rid="selectedRid" />
    <RecipeDocumentDialog v-model="docVisible" :initial-rid="selectedRid" />
    <RecipeUploadDialog v-model="uploadVisible" @uploaded="onUploaded" />
  </div>
</template>

<style scoped>
.browse-view {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
  height: 100%;
  overflow-y: auto;
}
.browse-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}
.browse-header h2 {
  margin: 0;
}
.count {
  color: var(--el-text-color-secondary);
  font-size: 14px;
}
.header-actions {
  margin-left: auto;
}
.filter-card {
  margin-bottom: 16px;
}
.filter-result {
  margin-top: 8px;
  font-size: 13px;
  color: var(--el-color-primary);
}
.recipe-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
}
.recipe-card {
  display: flex;
  flex-direction: column;
  transition: transform 0.15s;
}
.recipe-card:hover {
  transform: translateY(-2px);
}
.card-top {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.card-title {
  margin: 0;
  font-size: 16px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}
.card-meta {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 8px;
}
.meta-item {
  font-size: 13px;
}
.stars {
  color: #e6a23c;
  letter-spacing: 1px;
}
.card-desc {
  margin: 0 0 12px;
  font-size: 13px;
  color: var(--el-text-color-regular);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  line-height: 1.5;
  min-height: 39px;
}
.card-stats {
  display: flex;
  gap: 16px;
  font-size: 13px;
  color: var(--el-text-color-secondary);
  margin-bottom: 12px;
}
.card-actions {
  display: flex;
  gap: 6px;
  margin-top: auto;
}
</style>
