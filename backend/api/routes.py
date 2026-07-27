"""
FastAPI 路由定义 - 前后端分离接口

所有接口前缀 /api，复用 main.AdvancedGraphRAGSystem 引擎：
    GET  /api/health                  系统就绪状态（前端轮询）
    GET  /api/stats                    路由统计 + 知识库统计
    POST /api/query                    非流式问答
    POST /api/query/stream             SSE 流式问答（逐 token）
    POST /api/knowledge-base/rebuild   重建知识库
"""

import asyncio
import json
import logging
import time

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from .state import state
from .chat_history import conversation_store
from .schemas import (
    QueryRequest,
    QueryResponse,
    AnalysisInfo,
    SourceDoc,
    HealthResponse,
    RebuildResponse,
    ConversationCreateRequest,
    MessageCreateRequest,
    ConversationRenameRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

# 流式生成结束哨兵：在线程中取下一个 token 时，StopIteration 不能进入 asyncio
# Future（"interacts badly with generators"），故在线程内捕获并转为哨兵。
_STREAM_END = object()


def _next_chunk(gen):
    """在线程中取下一个流式片段，StopIteration 转为哨兵返回。"""
    try:
        return next(gen)
    except StopIteration:
        return _STREAM_END


def _require_system():
    """获取就绪的 RAG 系统，未就绪时抛 503。"""
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail=f"系统未就绪（{state.status}）"
            + (f"：{state.error}" if state.error else ""),
        )
    return state.system


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------
@router.get("/health", response_model=HealthResponse)
async def health():
    """返回系统就绪状态，前端启动时轮询此接口判断后端是否可用。"""
    message = None
    if state.status == "error":
        message = state.error
    elif state.status == "initializing":
        message = "RAG 系统初始化中，请稍候..."
    return HealthResponse(ready=state.ready, status=state.status, message=message)


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
@router.get("/stats")
async def stats():
    """返回路由统计 + 知识库统计 + Milvus 统计。"""
    system = _require_system()
    return system.get_system_stats()


# ---------------------------------------------------------------------------
# 知识图谱子图（可视化用）
# ---------------------------------------------------------------------------
@router.get("/knowledge-graph")
async def knowledge_graph(type: str, limit: int = 15, neighbor_limit: int = 6):
    """返回某类节点（recipes/ingredients/cooking_steps）的有界知识子图。

    Query 参数：
        type: recipes | ingredients | cooking_steps
        limit: 主节点数量上限（1-50，默认 15）
        neighbor_limit: 每主节点邻居数量上限（1-12，默认 6）
    """
    system = _require_system()
    try:
        return await asyncio.to_thread(
            system.get_knowledge_subgraph, type, limit, neighbor_limit
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# 非流式问答
# ---------------------------------------------------------------------------
@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """非流式问答：路由检索 + 一次性生成完整答案。"""
    system = _require_system()
    start = time.time()

    try:
        # 路由 + 检索（阻塞，放到线程池避免卡住事件循环）
        documents, analysis = await asyncio.to_thread(
            system.retrieve, req.question, req.top_k
        )
        analysis_dict = system.analysis_to_dict(analysis)
        sources = system.sources_from_documents(documents)

        if not documents:
            return QueryResponse(
                answer="抱歉，没有找到相关的烹饪信息。请尝试其他问题。",
                analysis=AnalysisInfo(**analysis_dict),
                sources=[SourceDoc(**s) for s in sources],
                elapsed=time.time() - start,
            )

        # 一次性生成完整答案（阻塞，放线程池）
        answer = await asyncio.to_thread(
            system.generation_module.generate_adaptive_answer, req.question, documents
        )

        return QueryResponse(
            answer=answer,
            analysis=AnalysisInfo(**analysis_dict),
            sources=[SourceDoc(**s) for s in sources],
            elapsed=time.time() - start,
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception("非流式问答失败")
        raise HTTPException(status_code=500, detail=f"处理问题时出现错误：{e}")


# ---------------------------------------------------------------------------
# SSE 流式问答
# ---------------------------------------------------------------------------
@router.post("/query/stream")
async def query_stream(req: QueryRequest):
    """SSE 流式问答。

    事件序列：
        1. event=analysis  data={"analysis": {...}, "sources": [...]}  （路由检索完成）
        2. event=chunk     data={"content": "..."}                     （逐 token，0..N 次）
        3. event=done      data={"elapsed": 12.34}                     （正常结束）
        或 event=error    data={"message": "..."}                      （任意阶段出错）
    """
    # 先校验就绪状态（SSE 开始后抛 HTTP 异常前端不易处理，提前校验）
    system = _require_system()

    question = req.question
    top_k = req.top_k

    async def event_generator():
        start = time.time()
        try:
            # 1. 路由 + 检索（阻塞，线程池）
            documents, analysis = await asyncio.to_thread(
                system.retrieve, question, top_k
            )
            yield {
                "event": "analysis",
                "data": json.dumps(
                    {
                        "analysis": system.analysis_to_dict(analysis),
                        "sources": system.sources_from_documents(documents),
                    },
                    ensure_ascii=False,
                ),
            }

            # 无检索结果：直接结束
            if not documents:
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {
                            "answer": "抱歉，没有找到相关的烹饪信息。请尝试其他问题。",
                            "elapsed": time.time() - start,
                        },
                        ensure_ascii=False,
                    ),
                }
                return

            # 2. 流式生成（同步生成器，逐 next 放线程池，避免阻塞事件循环）
            gen = system.generation_module.generate_adaptive_answer_stream(
                question, documents
            )
            while True:
                chunk = await asyncio.to_thread(_next_chunk, gen)
                if chunk is _STREAM_END:
                    break
                if chunk:
                    yield {
                        "event": "chunk",
                        "data": json.dumps({"content": chunk}, ensure_ascii=False),
                    }

            # 3. 正常结束
            yield {
                "event": "done",
                "data": json.dumps({"elapsed": time.time() - start}, ensure_ascii=False),
            }

        except Exception as e:
            logger.exception("流式问答失败")
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# 重建知识库
# ---------------------------------------------------------------------------
@router.post("/knowledge-base/rebuild", response_model=RebuildResponse)
async def rebuild_knowledge_base():
    """重建知识库（删除现有的向量数据并重新构建）。"""
    system = _require_system()
    result = await asyncio.to_thread(system.rebuild_knowledge_base)
    return RebuildResponse(**result)


# ---------------------------------------------------------------------------
# 聊天会话（多会话历史）
# ---------------------------------------------------------------------------
@router.get("/conversations")
async def list_conversations():
    """返回所有会话的元数据列表（不含消息全文），按最近更新倒序。"""
    return {"conversations": conversation_store.list_conversations()}


@router.post("/conversations")
async def create_conversation(req: ConversationCreateRequest):
    """新建一个空会话。"""
    conv = conversation_store.create_conversation(req.title)
    return {"conversation": conv}


@router.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """返回指定会话的完整内容（含 messages）。"""
    conv = conversation_store.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"conversation": conv}


@router.post("/conversations/{conv_id}/messages")
async def append_message(conv_id: str, req: MessageCreateRequest):
    """向指定会话追加一条消息。

    返回 {message, conversation}：conversation 为更新后的元数据，
    供前端刷新侧边栏（标题/更新时间/消息数/预览）。
    """
    result = conversation_store.append_message(
        conv_id=conv_id,
        role=req.role,
        content=req.content,
        analysis=req.analysis,
        sources=req.sources,
        elapsed=req.elapsed,
        error=req.error,
        timestamp=req.timestamp,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return result


@router.patch("/conversations/{conv_id}")
async def rename_conversation(conv_id: str, req: ConversationRenameRequest):
    """重命名会话。"""
    meta = conversation_store.rename_conversation(conv_id, req.title)
    if meta is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"conversation": meta}


@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """删除会话。"""
    ok = conversation_store.delete_conversation(conv_id)
    if not ok:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"ok": True}
