import MarkdownIt from 'markdown-it'

// 单例 markdown-it 实例：默认不渲染原始 HTML（安全），启用代码块/列表/表格
const md = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
  typographer: true,
})

/** 将 markdown 文本渲染为 HTML（用于展示 LLM 回答） */
export function renderMarkdown(text: string): string {
  if (!text) return ''
  return md.render(text)
}
