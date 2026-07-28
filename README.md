# 尝尝咸淡 · 图 RAG 智能烹饪助手

基于图数据库（Neo4j）+ 向量库（Milvus）+ LLM 的智能烹饪问答系统，采用 **前后端分离** 架构。

```
rag_modules/
├── backend/          后端工程（FastAPI + 原 RAG 引擎）
│   ├── app.py        FastAPI 入口（Web 服务）
│   ├── main.py       AdvancedGraphRAGSystem 编排层（保留 CLI 入口）
│   ├── api/          Web API 层（路由 / Schema / 状态 / 会话 / 评估）
│   ├── config.py     系统配置
│   ├── rag_modules/  核心 RAG 模块（数据/索引/检索/路由/生成/评估）
│   ├── data/         评估测试集（eval_dataset.json）
│   └── requirements.txt
└── frontend/         前端工程（Vue3 + Vite + Element Plus）
    └── src/
        ├── views/      聊天页 / 菜谱浏览 / 统计页 / 评估页
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
| GET | `/api/evaluation/status` | RAGAS 评估依赖是否可用 |
| GET | `/api/evaluation/dataset` | 内置烹饪评估测试集 |
| POST | `/api/evaluation/single` | 手动单条评估（question/answer/contexts[/ground_truth]） |
| POST | `/api/evaluation/message` | 评估会话中的某条问答（重新检索完整上下文） |
| POST | `/api/evaluation/run` | 运行测试集评估（完整 RAG 管线 + RAGAS） |
| GET | `/api/evaluation/results` | 历史评估列表 |
| GET | `/api/evaluation/results/{id}` | 单次评估详情 |
| DELETE | `/api/evaluation/results/{id}` | 删除评估记录 |

## RAGAS 评估

系统内置 [RAGAS](https://github.com/explodinggradients/ragas) 评估能力，量化 RAG 管线质量，复用现有 DeepSeek LLM + BGE 向量。

**四项核心指标：**
- **忠实度（Faithfulness）**：答案是否可由检索上下文支持（无幻觉），无需参考答案
- **答案相关性（ResponseRelevancy）**：答案是否切题
- **上下文召回率（ContextRecall）**：参考答案是否都能被检索上下文覆盖（需参考答案）
- **上下文精确率（ContextPrecision）**：相关检索项是否排在前面（需参考答案）

**两个入口（前端「评估」页）：**
- 测试集评估：对内置 8 条烹饪 Q&A 测试集（带参考答案）跑完整 RAG 管线 + 4 项指标，输出聚合均值 + 每样本明细
- 单条评估：从会话历史选一条助手回答评估，或手动粘贴 question/answer/contexts 评估

**可选依赖（不影响主应用）：**

```bash
cd backend
pip install ragas langchain-openai
```

RAGAS 为**可选依赖**：未安装时主应用与聊天/检索功能照常运行，仅评估接口返回 503 + 安装提示（前端评估页顶部会显示提示横幅）。

> judge 模型默认用系统 LLM，可用 `RAGAS_JUDGE_MODEL` 环境变量切换；并发数用 `RAGAS_MAX_WORKERS`（默认 3）。

> 后端 RAG 引擎与模块细节见 [backend/README.md](backend/README.md)。
