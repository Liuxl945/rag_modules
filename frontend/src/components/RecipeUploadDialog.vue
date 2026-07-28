<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import { uploadRecipe } from '@/api'
import type { UploadRecipeResponse } from '@/types'

const visible = defineModel<boolean>({ required: true })

const emit = defineEmits<{
  uploaded: [result: UploadRecipeResponse]
}>()

const fileList = ref<any[]>([])
const selectedFile = ref<File | null>(null)
const uploading = ref(false)
const error = ref('')
const result = ref<UploadRecipeResponse | null>(null)

function onFileChange(uploadFile: any) {
  error.value = ''
  result.value = null
  if (uploadFile.name && !uploadFile.name.toLowerCase().endsWith('.md')) {
    error.value = '只支持 .md 格式的 Markdown 文件'
    fileList.value = []
    selectedFile.value = null
    return
  }
  selectedFile.value = uploadFile.raw as File
}

function onFileRemove() {
  selectedFile.value = null
  error.value = ''
  result.value = null
}

async function doUpload() {
  if (!selectedFile.value) return
  uploading.value = true
  error.value = ''
  result.value = null
  try {
    const res = await uploadRecipe(selectedFile.value)
    result.value = res
    emit('uploaded', res)
    ElMessage.success(res.message)
  } catch (e: any) {
    error.value = e?.response?.data?.detail || '上传失败'
  } finally {
    uploading.value = false
  }
}

function reset() {
  fileList.value = []
  selectedFile.value = null
  error.value = ''
  result.value = null
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}
</script>

<template>
  <el-dialog
    v-model="visible"
    title="上传菜谱 (Markdown)"
    width="520px"
    :close-on-click-modal="false"
    @closed="reset"
  >
    <el-upload
      drag
      :auto-upload="false"
      accept=".md"
      :limit="1"
      :on-change="onFileChange"
      :on-remove="onFileRemove"
      :file-list="fileList"
    >
      <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
      <div class="el-upload__text">拖拽 .md 文件到此处，或<em>点击选择</em></div>
      <template #tip>
        <div class="el-upload__tip">
          仅支持 Markdown (.md) 格式的菜谱文件，最大 100KB
        </div>
      </template>
    </el-upload>

    <!-- 文件信息 -->
    <div v-if="selectedFile" class="file-info">
      <span>{{ selectedFile.name }}</span>
      <span class="file-size">{{ formatSize(selectedFile.size) }}</span>
    </div>

    <!-- 错误提示 -->
    <el-alert
      v-if="error"
      :title="error"
      type="error"
      :closable="false"
      show-icon
      style="margin-top: 12px"
    />

    <!-- 成功提示 -->
    <el-alert
      v-if="result"
      :title="result.message"
      type="success"
      :closable="false"
      show-icon
      style="margin-top: 12px"
    >
      <template #default>
        <div>{{ result.message }}</div>
        <div v-if="result.recipe_name" class="result-detail">
          <span>菜谱: {{ result.recipe_name }}</span>
          <span>食材: {{ result.ingredients }}</span>
          <span>步骤: {{ result.steps }}</span>
          <span>文本块: {{ result.chunks_created }}</span>
        </div>
      </template>
    </el-alert>

    <template #footer>
      <el-button @click="visible = false">关闭</el-button>
      <el-button v-if="result" type="primary" @click="reset">继续上传</el-button>
      <el-button
        v-else
        type="primary"
        :loading="uploading"
        :disabled="!selectedFile || !!error"
        @click="doUpload"
      >
        上传
      </el-button>
    </template>
  </el-dialog>
</template>

<style scoped>
.file-info {
  margin-top: 10px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 13px;
  color: var(--el-text-color-regular);
}
.file-size {
  color: var(--el-text-color-secondary);
}
.result-detail {
  margin-top: 4px;
  font-size: 13px;
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}
</style>
