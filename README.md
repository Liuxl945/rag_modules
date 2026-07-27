# 尝尝咸淡 · 图 RAG 智能烹饪助手

基于图数据库（Neo4j）+ 向量库（Milvus）+ LLM 的智能烹饪问答系统，采用 **前后端分离** 架构。

```
rag_modules/
├── backend/          后端工程（FastAPI + 原 RAG 引擎）
│   ├── app.py        FastAPI 入口（Web 服务）
│   ├── main.py       AdvancedGraphRAGSystem 编排层（保留 CLI 入口）
│   ├── api/          Web API 层（路由 / Schema / 状态）
│   ├── config.py     系统配置
│   ├── rag_modules/  核心 RAG 模块（数据/索引/检索/路由/生成）
│   └── requirements.txt
└── frontend/         前端工程（Vue3 + Vite + Element Plus）
    └── src/
        ├── views/      聊天页 / 统计页
        ├── components/ 消息 / 输入 / 来源 / 分析标签
        ├── api/        接口封装
        ├── utils/      SSE 流式客户端 / markdown 渲染
        └── stores/     Pinia 系统状态
```

## 架构

- **后端**：`backend/app.py` 启动 FastAPI 服务，后台异步初始化 `AdvancedGraphRAGSystem`（连 Neo4j/Milvus、加载模型、构建知识库）。通过 REST + SSE 接口对外提供问答、统计、重建能力，复用 `rag_modules/` 全部检索/生成逻辑。
- **前端**：Vue3 单页应用。聊天页通过 SSE 流式逐字显示 LLM 回答，并展示路由分析与检索来源；统计页展示知识库与路由统计，支持一键重建。

## 快速开始

### 1. 准备环境
- Neo4j（默认 `bolt://localhost:17687`）与 Milvus（默认 `localhost:19530`）已运行
- `backend/.env` 中配置 `DEEPSEEK_API_KEY`（参考 `.env.example`）

### 2. 启动后端
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
# 或保留原 CLI 交互模式：python main.py
```
启动后访问 `http://localhost:8000/docs` 查看接口文档。系统初始化需一定时间，`/api/health` 返回 `ready=true` 后即可查询。

### 3. 启动前端
```bash
cd frontend
npm install
npm run dev
```
访问 `http://localhost:5173`，开发环境已配置 `/api` 代理到后端 8000 端口。

## API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 系统就绪状态 |
| GET | `/api/stats` | 路由统计 + 知识库统计 |
| POST | `/api/query` | 非流式问答 |
| POST | `/api/query/stream` | SSE 流式问答（逐 token） |
| POST | `/api/knowledge-base/rebuild` | 重建知识库 |

> 后端 RAG 引擎与模块细节见 [backend/README.md](backend/README.md)。
