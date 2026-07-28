"""
Pydantic 请求/响应模型 - 定义前后端数据契约
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """问答请求"""

    question: str = Field(..., min_length=1, description="用户的问题")
    top_k: Optional[int] = Field(None, ge=1, description="返回结果数量上限（留空用默认值）")


class SourceDoc(BaseModel):
    """检索来源摘要"""

    recipe_name: str
    search_type: str
    score: float
    content_preview: str


class AnalysisInfo(BaseModel):
    """路由分析结果"""

    recommended_strategy: str
    query_complexity: float
    relationship_intensity: float
    reasoning_required: bool
    entity_count: int
    confidence: float
    reasoning: str


class QueryResponse(BaseModel):
    """非流式问答响应"""

    answer: str
    analysis: AnalysisInfo
    sources: List[SourceDoc]
    elapsed: float


class HealthResponse(BaseModel):
    """系统就绪状态"""

    ready: bool
    status: str = Field(..., description="initializing | ready | error")
    message: Optional[str] = None


class RebuildResponse(BaseModel):
    """知识库重建响应"""

    success: bool
    message: str
    stats: dict


class UploadRecipeResponse(BaseModel):
    """Markdown 菜谱上传响应"""

    success: bool
    message: str
    recipe_id: Optional[str] = None
    recipe_name: Optional[str] = None
    chunks_created: Optional[int] = None
    ingredients: Optional[int] = None
    steps: Optional[int] = None
    milvus_ok: Optional[bool] = None
    stats: Optional[dict] = None


class ErrorResponse(BaseModel):
    """错误响应"""

    detail: str


# ---------------------------------------------------------------------------
# 聊天会话（多会话历史）
# ---------------------------------------------------------------------------
class ConversationCreateRequest(BaseModel):
    """新建会话请求（标题可选，缺省由后端取「新对话」占位）"""

    title: Optional[str] = Field(None, description="会话标题，留空则用占位标题")


class MessageCreateRequest(BaseModel):
    """向会话追加一条消息"""

    role: str = Field(..., description="user | assistant")
    content: str = Field(..., description="消息内容")
    analysis: Optional[dict] = None
    sources: Optional[List[dict]] = None
    elapsed: Optional[float] = None
    error: Optional[bool] = False
    timestamp: Optional[float] = None


class ConversationRenameRequest(BaseModel):
    """重命名会话"""

    title: str = Field(..., min_length=1, description="新标题")
