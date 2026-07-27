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


class ErrorResponse(BaseModel):
    """错误响应"""

    detail: str
