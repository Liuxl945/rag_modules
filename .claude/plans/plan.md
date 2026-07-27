# 前后端分离改造方案

## 目标
把当前的 CLI 式 Graph RAG 烹饪助手改造成前后端分离项目：
- `backend/` —— 后端工程，新增 FastAPI Web 服务（保留原 CLI 入口）
- `frontend/` —— 前端工程，Vue3 + Vite + Element Plus 聊天界面

不改动 `rag_modules/` 核心检索/生成模块，只在外层加 Web 编排与前端。

---

## 一、后端改造（FastAPI）

### 1. 依赖（`backend/requirements.txt` 追加）
- `fastapi>=0.110.0`
- `uvicorn[standard]>=0.27.0`
- `sse-starlette>=1.6.0`（SSE 流式事件）
- pydantic 已存在

### 2. 重构 `backend/main.py` 的 `AdvancedGraphRAGSystem`
保留原 CLI（`run_interactive` / `ask_question_with_routing` / `main()`）不变，新增「数据返回型」方法供 API 调用（不打 print、不读 input）：
- `retrieve(question, top_k=None) -> (documents, analysis)`：仅做路由+检索（封装 `query_router.route_query`），未就绪抛 `ValueError`
- `get_system_stats() -> dict`：返回路由统计 + 知识库统计 + Milvus 统计 + ready 标记
- `rebuild_knowledge_base() -> dict`：**无确认 prompt** 的重建（从 `_rebuild_knowledge_base` 抽出，原交互方法保留确认后委托给它）
- `analysis_to_dict(analysis)`：把 `QueryAnalysis`（含 `SearchStrategy` 枚举）序列化为可 JSON 化的 dict
- `sources_from_documents(docs)`：从 Document.metadata 提取来源信息（recipe_name / search_type / score / route_strategy）
- 生成直接复用 `generation_module.generate_adaptive_answer[_stream]`，不再包裹

### 3. 新增 `backend/app.py`（FastAPI 应用）
- **lifespan**：启动时后台异步初始化 RAG 系统（`initialize_system` + `build_knowledge_base` 用 `asyncio.to_thread` 跑，不阻塞事件循环）；维护全局 `AppState`（status: `initializing` / `ready` / `error`）；关闭时调 `_cleanup()`
- **CORS**：放行 Vite 默认端口 `http://localhost:5173`（可由 `CORS_ORIGINS` 环境变量覆盖）
- 系统未就绪时，需系统的接口返回 `503` + 明确 message

### 4. 新增 `backend/api/schemas.py`（Pydantic 模型）
`QueryRequest` / `SourceDoc` / `AnalysisInfo` / `QueryResponse` / `HealthResponse` / `StatsResponse` / `RebuildResponse`

### 5. 新增 `backend/api/routes.py`（路由）
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 系统就绪状态（前端轮询用） |
| GET | `/api/stats` | 路由统计 + 知识库统计 |
| POST | `/api/query` | 非流式问答，返回 `{answer, analysis, sources, elapsed}` |
| POST | `/api/query/stream` | **SSE 流式**：先发 `analysis`+`sources` 事件，再逐 token 发 `chunk`，最后 `done`（出错发 `error`） |
| POST | `/api/knowledge-base/rebuild` | 重建知识库 |

SSE 用 `sse-starlette` 的 `EventSourceResponse`，生成器内 `yield` 命名事件。

### 6. 运行方式
`cd backend && uvicorn app:app --reload --port 8000`（`python main.py` 仍可走原 CLI）

---

## 二、前端工程（Vue3 + Vite + Element Plus）

### 1. 脚手架与依赖
- Vite + Vue3 + TypeScript
- `vue-router`、`pinia`、`axios`、`element-plus`、`markdown-it`（渲染 LLM 的 markdown 回答）、`@types/markdown-it`

### 2. `vite.config.ts`
开发代理：`/api` -> `http://localhost:8000`

### 3. 目录结构
```
frontend/
  index.html
  package.json / vite.config.ts / tsconfig.json
  src/
    main.ts            # createApp + ElementPlus + router + pinia
    App.vue            # 布局 + 顶部状态条（连接/初始化状态）
    router/index.ts    # / 聊天, /stats 统计
    api/index.ts       # axios 实例 + 接口封装
    stores/system.ts   # pinia: 就绪状态、统计、轮询 health
    utils/sse.ts       # fetch+ReadableStream 手写 SSE 解析（POST 流式）
    views/
      ChatView.vue     # 聊天主界面
      StatsView.vue    # 统计面板 + 重建按钮
    components/
      ChatMessage.vue  # 单条消息（用户/助手，markdown 渲染）
      MessageInput.vue # 输入框 + 发送
      SourceList.vue   # 检索来源（可折叠）
      AnalysisTag.vue  # 路由策略/复杂度标签
    assets/main.css
```

### 4. 聊天界面功能
- 消息列表（用户靠右、助手靠左），助手消息流式逐字显示
- 每条助手消息下方可折叠展示：路由分析（策略标签、复杂度、关系密集度、置信度）+ 检索来源列表（菜谱名、检索类型、得分）
- 输入框 Enter 发送、发送中禁用、流式中可「停止」
- 顶部状态条：后端初始化中 / 就绪 / 异常（轮询 `/api/health`）

### 5. 统计面板
- 知识库统计（菜谱/食材/步骤/文档/分块/向量数）
- 路由统计（三策略次数 + 占比，Element Plus 进度条）
- 「重建知识库」按钮（`ElMessageBox.confirm` 确认后调用，显示 loading）

### 6. 运行方式
`cd frontend && npm install && npm run dev`（默认 5173）

---

## 三、其他
- 根 `README.md`：补充前后端分离启动说明
- `.gitignore`：追加 `frontend/node_modules/`、`frontend/dist/`、`*.local`
- 不动 `rag_modules/` 内部逻辑、不动 `config.py`、不动 `.env`

## 四、验证范围
- 后端：`python -c "import app"` 可导入（FastAPI 应用定义层）；完整运行需 Neo4j/Milvus/DeepSeek，无法在此环境实测，但代码保证语法/导入正确、接口契约清晰
- 前端：`npm install` + `npm run build` 通过编译
