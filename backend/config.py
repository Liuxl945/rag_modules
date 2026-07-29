"""
基于图数据库的RAG系统配置文件
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class GraphRAGConfig:
    """基于图数据库的RAG系统配置类"""

    # Neo4j数据库配置
    neo4j_uri: str = "bolt://localhost:17687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "all-in-rag"
    neo4j_database: str = "neo4j"

    # Milvus配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "cooking_knowledge"
    milvus_dimension: int = 512  # BGE-small-zh-v1.5的向量维度

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "deepseek-v4-flash"

    # 检索配置（LightRAG Round-robin策略）
    # top_k 收敛到 3：减少喂给 LLM 的文档块数量以降低噪声、提升精确率。
    # 配合 enable_rerank 精排，头部质量更高，减少返回数量不会丢关键信息。
    top_k: int = 5

    # 重排序（Rerank）配置 - cross-encoder 对初检（RRF 融合）结果精排，把最相关文档排前面再取 top_k。
    # 标准 two-stage retrieval：召回较多候选 -> cross-encoder 精排 -> 只留 top_k。
    enable_rerank: bool = True                     # 默认开启；模型未缓存时自动降级（跳过重排，返回 RRF 顺序）
    rerank_model: str = "BAAI/bge-reranker-v2-m3"  # 中文 cross-encoder 重排器（首次需下载，见 scripts/download_reranker.py）
    rerank_candidate_k: int = 20                    # 送入重排的候选池大小（应 >= top_k；越大越准但越慢）
    rerank_max_length: int = 512                    # cross-encoder 单对 (query, doc) 最大 token 数

    # 父文档检索配置
    # 开启后，RRF+rerank 排前 N 的 chunk 会被替换为完整菜谱文档（含食材+步骤），
    # 避免只检索到"标签"等局部 chunk 导致 LLM 只能从关键词编答案。
    enable_parent_doc_retrieval: bool = True
    parent_doc_top_n: int = 5                   # 前 N 名做父文档替换（与 top_k=5 对齐，确保返回的都是完整菜谱）
    parent_doc_max_chars: int = 4000            # 每篇父文档字符上限（兜底）

    # BM25 分词缓存配置
    # jieba 分词是 CPU 密集操作，文档量大时启动耗时主要花在分词上。
    # 将 tokenized_corpus 持久化到磁盘，启动时优先读缓存，命中则跳过分词。
    # 留空 -> 使用 hybrid_retrieval.py 同级目录下的 .bm25_cache/
    bm25_cache_dir: str = ""

    # 生成配置（控制 LLM 回答的随机性和长度）
    temperature: float = 0.1        # 低温度 → 回答更确定、更稳定
    max_tokens: int = 2048          # 单次回答最大输出 token 数

    # combined 策略下的图 RAG 辅助预算
    # combined 检索中，图 RAG 返回的是「多跳路径/知识子图」等推理线索，不是答案主体证据；
    # 因此只给少量名额（默认 2），避免空壳子图或无关路径占用 top_k 挤掉高质量菜谱。
    # 传统路始终分配完整 top_k，让 reranker 从候选池里自由挑最优结果。
    combined_graph_budget: int = 2

    # 图数据处理配置（文档分块与图遍历）
    chunk_size: int = 500           # 文档分块的目标大小（字符数）
    chunk_overlap: int = 50         # 相邻块的 overlap，避免切分打断上下文
    max_graph_depth: int = 2        # 图遍历最大跳数（控制推理深度）

    def __post_init__(self):
        """初始化后的处理"""
        # LightRAG使用Round-robin策略，无需权重验证
        pass
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GraphRAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'neo4j_uri': self.neo4j_uri,
            'neo4j_user': self.neo4j_user,
            'neo4j_password': self.neo4j_password,
            'neo4j_database': self.neo4j_database,
            'milvus_host': self.milvus_host,
            'milvus_port': self.milvus_port,
            'milvus_collection_name': self.milvus_collection_name,
            'milvus_dimension': self.milvus_dimension,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'enable_rerank': self.enable_rerank,
            'rerank_model': self.rerank_model,
            'rerank_candidate_k': self.rerank_candidate_k,
            'rerank_max_length': self.rerank_max_length,
            'enable_parent_doc_retrieval': self.enable_parent_doc_retrieval,
            'parent_doc_top_n': self.parent_doc_top_n,
            'parent_doc_max_chars': self.parent_doc_max_chars,
            'bm25_cache_dir': self.bm25_cache_dir,

            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'combined_graph_budget': self.combined_graph_budget,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_graph_depth': self.max_graph_depth
        }

# 默认配置实例
DEFAULT_CONFIG = GraphRAGConfig() 