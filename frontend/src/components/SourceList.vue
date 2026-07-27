<script setup lang="ts">
import type { SourceDoc } from '@/types'

defineProps<{ sources: SourceDoc[] }>()
</script>

<template>
  <el-collapse v-if="sources.length" class="source-list">
    <el-collapse-item :title="`检索来源（${sources.length}）`" name="sources">
      <div v-for="(src, i) in sources" :key="i" class="source-item">
        <div class="source-head">
          <span class="source-name">{{ src.recipe_name }}</span>
          <el-tag size="small" type="info">{{ src.search_type }}</el-tag>
          <span class="source-score">得分 {{ src.score.toFixed(3) }}</span>
        </div>
        <div class="source-preview">{{ src.content_preview }}</div>
      </div>
    </el-collapse-item>
  </el-collapse>
</template>

<style scoped>
.source-list {
  margin-top: 8px;
}
.source-item {
  padding: 6px 0;
  border-bottom: 1px dashed var(--el-border-color-lighter);
}
.source-item:last-child {
  border-bottom: none;
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
}
.source-preview {
  margin-top: 4px;
  color: var(--el-text-color-regular);
  font-size: 13px;
  line-height: 1.5;
  white-space: pre-wrap;
}
</style>
