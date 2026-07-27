"""
FastAPI 应用入口 - 图 RAG 智能烹饪助手 Web 服务

启动方式：
    cd backend
    uvicorn app:app --reload --port 8000

架构：
    lifespan 启动时后台异步初始化 AdvancedGraphRAGSystem（不阻塞 HTTP 服务），
    /api/health 反映就绪状态，未就绪时业务接口返回 503。
    原 CLI 入口 main.py 的 python main.py 仍可使用。
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from main import AdvancedGraphRAGSystem
from api.routes import router
from api.state import state

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def _init_rag_system():
    """后台初始化 RAG 系统。

    initialize_system / build_knowledge_base 是阻塞操作（连 Neo4j/Milvus、
    加载嵌入模型、构建索引），放到线程池执行，避免卡住事件循环导致 /api/health 无响应。
    """
    try:
        logger.info("开始后台初始化 RAG 系统...")
        system = AdvancedGraphRAGSystem()
        await asyncio.to_thread(system.initialize_system)
        await asyncio.to_thread(system.build_knowledge_base)
        state.system = system
        state.status = "ready"
        state.error = None
        logger.info("✅ RAG 系统初始化完成，Web 服务就绪")
    except Exception as e:
        state.status = "error"
        state.error = str(e)
        logger.exception("❌ RAG 系统初始化失败")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时后台初始化，关闭时清理资源。"""
    init_task = asyncio.create_task(_init_rag_system())
    try:
        yield
    finally:
        # 关闭：初始化仍在进行则取消；已完成则清理数据库连接
        if not init_task.done():
            init_task.cancel()
            try:
                await init_task
            except (asyncio.CancelledError, Exception):
                pass
        elif state.system is not None and state.ready:
            try:
                await asyncio.to_thread(state.system._cleanup)
            except Exception as e:
                logger.warning(f"清理资源失败: {e}")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    app = FastAPI(
        title="图 RAG 智能烹饪助手 API",
        description="基于图 RAG 的智能烹饪助手后端接口（前后端分离）",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS：默认放行 Vite 开发服务器（5173），可用 CORS_ORIGINS 环境变量覆盖
    default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
    env_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
    cors_origins = env_origins or default_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/", tags=["meta"])
    async def root():
        return {"name": "图 RAG 智能烹饪助手 API", "status": state.status, "docs": "/docs"}

    return app


app = create_app()


if __name__ == "__main__":
    # 便于直接 python app.py 启动（开发用）
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
