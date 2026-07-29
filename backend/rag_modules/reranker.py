"""
重排序模块 - 基于 cross-encoder 的精排

对初检（RRF 融合）结果用 cross-encoder 重新打分排序，提升 top-k 精确率。
cross-encoder 将 (query, doc) 作为一对同时输入模型，输出相关性分数，
比 bi-encoder（query/doc 各自编码后算余弦）更能捕捉细粒度的语义相关性。

两阶段检索（two-stage retrieval）：
    召回（BM25+向量+图键值, RRF 融合）-> 取较多候选（rerank_candidate_k）
        ↓
    cross-encoder 精排（本模块）-> 取 top_k

设计要点（沿用项目既有模式：懒加载 + 可降级）：
    - 模型首次 rerank() 调用时才加载（启动零成本，与 evaluation 模块一致）
    - 模型未缓存 / 加载失败 / 打分异常时优雅降级：跳过重排，返回原 RRF 顺序
    - _ensure_model() 加锁，防止并发首查重复加载

外部依赖：sentence_transformers.CrossEncoder（已在 requirements.txt）

模型：BAAI/bge-reranker-v2-m3（中文 cross-encoder，~568MB）
    首次需下载（HF_HUB_OFFLINE=1 时无法联网下载）：
        HF_HUB_OFFLINE=0 python scripts/download_reranker.py
      或用国内镜像：
        HF_ENDPOINT=https://hf-mirror.com python scripts/download_reranker.py
    下载后永久缓存，之后离线模式也能用。
"""

import logging
import math
import threading
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RerankerModule:
    """cross-encoder 重排序模块（懒加载、可降级）。

    Public API:
        - available: bool 属性，触发懒加载并返回模型是否就绪
        - rerank(query, documents, top_k) -> 精排后取 top_k 的 Document 列表
    """

    def __init__(self, config):
        """仅存配置，不加载模型。

        Args:
            config: GraphRAGConfig 配置对象（读取 rerank_model / rerank_max_length 等）
        """
        self.config = config
        self._model = None            # CrossEncoder 实例（懒加载）
        self._load_attempted = False  # 是否已尝试过加载（避免重复尝试）
        self._available = False       # 模型是否就绪
        self._lock = threading.Lock()  # 保护懒加载，防并发首查重复加载

    def _ensure_model(self) -> bool:
        """懒加载重排模型（线程安全）。仅在首次调用时加载。

        Returns:
            True -> 模型就绪可重排；False -> 加载失败（调用方应降级跳过重排）
        """
        if self._load_attempted:
            return self._available

        with self._lock:
            # double-check：拿到锁后可能已被另一线程加载完成
            if self._load_attempted:
                return self._available

            self._load_attempted = True
            model_name = getattr(self.config, "rerank_model", "BAAI/bge-reranker-v2-m3")
            max_length = getattr(self.config, "rerank_max_length", 512)

            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"加载重排模型: {model_name}（首次加载约 10-30s）...")
                self._model = CrossEncoder(
                    model_name,
                    max_length=max_length,
                    device="cpu",  # 与嵌入模型一致，使用 CPU 推理
                )
                self._available = True
                logger.info(f"✅ 重排模型加载完成: {model_name}")
            except Exception as e:
                # 模型未缓存（HF_HUB_OFFLINE=1）/ 依赖缺失 / 加载异常 -> 降级
                logger.warning(
                    f"重排模型加载失败，将跳过重排（返回 RRF 原顺序）: {e}\n"
                    f"如需启用重排，请先下载模型：\n"
                    f"  HF_HUB_OFFLINE=0 python scripts/download_reranker.py\n"
                    f"  或 HF_ENDPOINT=https://hf-mirror.com python scripts/download_reranker.py"
                )
                self._available = False
                self._model = None

            return self._available

    @property
    def available(self) -> bool:
        """模型是否就绪（触发懒加载）。"""
        return self._ensure_model()

    @staticmethod
    def _sigmoid(x: float) -> float:
        """将 cross-encoder 原始 logit 归一化到 (0,1)，便于展示且保持单调性。"""
        try:
            return 1.0 / (1.0 + math.exp(-float(x)))
        except (OverflowError, TypeError, ValueError):
            # 极大/极小值溢出时钳制
            return 1.0 if x and float(x) > 0 else 0.0

    def rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """对 documents 用 cross-encoder 重打分，按分数降序取 top_k。

        Args:
            query: 用户查询文本
            documents: 初检（RRF 融合）候选文档列表
            top_k: 最终返回数量

        Returns:
            精排后的 Document 列表（前 top_k 个）。每个 doc 的 metadata 新增：
                - rerank_score: 归一化后的重排分 (0,1)
                - final_score: 等于 rerank_score（重排后以精排分为最终分，使"得分"与排序一致）
                - reranked: True（标记经过重排）
              原 rrf_score 等融合分保留不变。

        降级：模型不可用 / 打分异常 / 空候选时，返回原顺序前 top_k 个（不重排）。
        """
        if not documents:
            return documents

        if not self._ensure_model():
            # 模型不可用：降级，保持 RRF 原顺序
            return documents[:top_k]

        # 构造 (query, doc) 文本对，送入 cross-encoder 打分
        pairs = [(query, d.page_content) for d in documents]
        try:
            scores = self._model.predict(
                pairs, batch_size=16, show_progress_bar=False
            )
        except Exception as e:
            logger.warning(f"重排打分失败，返回 RRF 原顺序: {e}")
            return documents[:top_k]

        # 按重排分降序，取 top_k
        ranked = sorted(
            zip(documents, scores), key=lambda x: float(x[1]), reverse=True
        )

        out: List[Document] = []
        for doc, raw_score in ranked[:top_k]:
            rerank_score = self._sigmoid(raw_score)
            new_metadata = dict(doc.metadata)  # 浅 copy，不 mutate 上游 Document
            new_metadata["rerank_score"] = rerank_score       # 归一化重排分 (0,1)
            new_metadata["final_score"] = rerank_score       # 重排后最终分 = 精排分
            new_metadata["reranked"] = True                  # 标记经过重排
            # rrf_score 等融合分保留在新_metadata 中（来自原 doc），供对比面板展示
            out.append(Document(page_content=doc.page_content, metadata=new_metadata))

        logger.info(
            f"重排完成：候选 {len(documents)} -> 取 top {len(out)}，"
            f"首条 rerank_score={out[0].metadata['rerank_score']:.4f}" if out else "重排完成：空结果"
        )
        return out
