"""
Pydantic 请求/响应模型 - 定义前后端数据契约
"""

from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """问答请求"""

    question: str = Field(..., min_length=1, description="用户的问题")
    top_k: Optional[int] = Field(None, ge=1, description="返回结果数量上限（留空用默认值）")


class SourceDoc(BaseModel):
    """检索来源摘要（含 chunk 元信息与各通道得分，供前端可视化）"""

    # 基础信息
    recipe_name: str
    search_type: str
    search_method: Optional[str] = None
    score: float
    final_score: Optional[float] = None
    content_preview: str
    # chunk / 文档定位
    node_id: Optional[str] = None
    chunk_id: Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    section_title: Optional[str] = None
    # 各检索通道命中情况与得分对比
    rrf_sources: Optional[List[str]] = None
    rrf_ranks: Optional[Dict[str, int]] = None
    rrf_raw_scores: Optional[Dict[str, float]] = None
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    dual_score: Optional[float] = None
    # 图 RAG 路径元信息
    path_length: Optional[int] = None
    node_count: Optional[int] = None
    relationship_count: Optional[int] = None


class AnalysisInfo(BaseModel):
    """路由分析结果"""

    recommended_strategy: str
    query_complexity: float
    relationship_intensity: float
    reasoning_required: bool
    entity_count: int
    confidence: float
    reasoning: str


class GraphQueryPlan(BaseModel):
    """图 RAG 查询规划（走了哪些实体/关系、最大跳数）"""

    query_type: str
    source_entities: List[str] = []
    target_entities: List[str] = []
    relation_types: List[str] = []
    max_depth: int = 2


class GraphPathNode(BaseModel):
    """图路径上的节点"""

    name: str = ""
    labels: Optional[List[str]] = None


class GraphPathRel(BaseModel):
    """图路径上的关系"""

    type: str = ""


class GraphPath(BaseModel):
    """图推理路径（节点链 + 关系链 + 跳数 + 相关性分）"""

    type: str  # graph_path | knowledge_subgraph
    recipe_name: str
    path_length: Optional[int] = None
    relevance_score: Optional[float] = None
    nodes: List[GraphPathNode] = []
    relationships: List[GraphPathRel] = []
    node_count: Optional[int] = None
    relationship_count: Optional[int] = None
    graph_density: Optional[float] = None
    reasoning_chains: List[str] = []


class ChannelStats(BaseModel):
    """三路召回（dual_level / vector / bm25）统计"""

    candidates: Dict[str, int] = {}     # 各路候选数（融合前）
    final: int = 0                       # 融合后入选数
    channels: List[str] = []             # 通道顺序
    contributed: Dict[str, int] = {}     # 各通道在最终结果中的入选数


class RetrievalTrace(BaseModel):
    """检索过程轨迹：为什么推荐这些结果"""

    graph_query_plan: Optional[GraphQueryPlan] = None
    graph_paths: List[GraphPath] = []
    channel_stats: Optional[ChannelStats] = None


class QueryResponse(BaseModel):
    """非流式问答响应"""

    answer: str
    analysis: AnalysisInfo
    sources: List[SourceDoc]
    elapsed: float
    retrieval_trace: Optional[RetrievalTrace] = None


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
    retrieval_trace: Optional[dict] = None
    elapsed: Optional[float] = None
    error: Optional[bool] = False
    timestamp: Optional[float] = None


class ConversationRenameRequest(BaseModel):
    """重命名会话"""

    title: str = Field(..., min_length=1, description="新标题")


# ---------------------------------------------------------------------------
# RAGAS 评估
# ---------------------------------------------------------------------------
class EvaluationSampleIn(BaseModel):
    """手动单条评估输入"""

    question: str = Field(..., min_length=1, description="用户问题")
    answer: str = Field(..., description="待评估的回答")
    contexts: List[str] = Field(default_factory=list, description="检索上下文片段列表")
    ground_truth: Optional[str] = Field(None, description="参考答案（提供则跑完整 4 指标）")


class EvaluationSingleResponse(BaseModel):
    """单条评估响应（scores 为 指标列名 -> 得分）"""

    scores: Dict[str, Optional[float]] = {}
    metrics: List[str] = []
    count: int = 1
    elapsed: Optional[float] = None


class MessageEvaluationRequest(BaseModel):
    """评估会话中的某条问答（重新检索取完整 contexts）"""

    conversation_id: str = Field(..., description="会话 ID")
    message_id: str = Field(..., description="助手消息 ID")


class EvaluationRunRequest(BaseModel):
    """运行测试集评估（questions 留空用内置测试集）"""

    questions: Optional[List[str]] = Field(
        None, description="自定义问题列表；留空则用内置烹饪测试集"
    )


class EvaluationRunResponse(BaseModel):
    """测试集评估响应"""

    run_id: str
    results: List[dict] = []
    aggregates: Dict[str, Optional[float]] = {}
    metrics: List[str] = []
    count: int = 0
    elapsed: Optional[float] = None
    skipped: int = 0


class EvaluationRunSummary(BaseModel):
    """评估运行元数据（列表项，不含 results 全文）"""

    id: str
    created_at: float
    kind: str
    count: int = 0
    metrics: List[str] = []
    aggregates: Dict[str, Optional[float]] = {}
    elapsed: Optional[float] = None


class EvaluationDatasetItem(BaseModel):
    """内置测试集条目"""

    id: str
    question: str
    ground_truth: str
    category: Optional[str] = None
