"""
RAGAS 评估模块 - 基于 RAGAS 库对 RAG 管线做质量评估

四项核心指标：
    - Faithfulness（忠实度）：答案是否可由检索上下文支持（无幻觉），无需 ground_truth
    - ResponseRelevancy（答案相关性）：答案是否切题，无需 ground_truth（需 embeddings）
    - LLMContextRecall（上下文召回率）：参考答案是否都能被检索上下文覆盖，需 ground_truth
    - LLMContextPrecisionWithReference（上下文精确率）：相关项是否排在前面（MRR 风格），需 ground_truth

设计要点：
    - 懒加载：ragas / LLM / embeddings 均在首次评估时初始化，系统启动零成本
    - 可降级：ragas 未安装时标记不可用，evaluate_* 抛 RuntimeError，由路由层捕获返回 503，
      不影响主应用与聊天/检索功能
    - 复用基础设施：LLM 用 DeepSeek（OpenAI 兼容），embeddings 复用 BGE
    - 防御式导入：ragas 0.2 指标名优先，0.1 别名回退，适配版本差异

外部依赖：
    - ragas>=0.2.0（评估框架）
    - langchain-openai（ChatOpenAI 封装 DeepSeek）
    - langchain-huggingface（BGE embeddings，已在主依赖中）
"""

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 防御式导入 ragas：未安装时标记不可用，主应用不受影响
# ---------------------------------------------------------------------------
_RAGAS_AVAILABLE = False
_RAGAS_IMPORT_ERROR: Optional[str] = None
_Faithfulness = None
_ResponseRelevancy = None
_LLMContextRecall = None
_LLMContextPrecisionWithReference = None
_SingleTurnSample = None
_EvaluationDataset = None
_evaluate = None
_LangchainLLMWrapper = None
_LangchainEmbeddingsWrapper = None
_RunConfig = None

try:
    from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig

    # 指标名防御式导入：0.2 主名优先，0.1 别名回退
    try:
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
            LLMContextRecall,
            LLMContextPrecisionWithReference,
        )
    except ImportError:  # pragma: no cover - ragas 0.1 兼容回退
        from ragas.metrics import Faithfulness
        from ragas.metrics import AnswerRelevancy as ResponseRelevancy
        from ragas.metrics import ContextRecall as LLMContextRecall
        from ragas.metrics import ContextPrecision as LLMContextPrecisionWithReference

    _SingleTurnSample = SingleTurnSample
    _EvaluationDataset = EvaluationDataset
    _evaluate = evaluate
    _LangchainLLMWrapper = LangchainLLMWrapper
    _LangchainEmbeddingsWrapper = LangchainEmbeddingsWrapper
    _RunConfig = RunConfig
    _Faithfulness = Faithfulness
    _ResponseRelevancy = ResponseRelevancy
    _LLMContextRecall = LLMContextRecall
    _LLMContextPrecisionWithReference = LLMContextPrecisionWithReference
    _RAGAS_AVAILABLE = True
except ImportError as e:  # ragas 未安装
    _RAGAS_IMPORT_ERROR = str(e)


def ragas_available() -> bool:
    """RAGAS 是否已安装可用（供路由层 / 健康检查查询，避免触发懒加载）。"""
    return _RAGAS_AVAILABLE


def _require_ragas():
    """ragas 未安装时抛带清晰安装提示的 RuntimeError，供路由层捕获返回 503。"""
    if not _RAGAS_AVAILABLE:
        raise RuntimeError(
            "RAGAS 未安装。请先安装评估依赖：pip install ragas langchain-openai"
            + (f"（导入错误：{_RAGAS_IMPORT_ERROR}）" if _RAGAS_IMPORT_ERROR else "")
        )


# ---------------------------------------------------------------------------
# 指标元信息（中文标签 + 是否需 ground_truth），供前后端一致展示
# ---------------------------------------------------------------------------
# key = RAGAS 结果 DataFrame 的列名
METRIC_META: Dict[str, Dict[str, Any]] = {
    "faithfulness": {"label": "忠实度", "needs_reference": False,
                     "desc": "答案是否可由检索上下文支持（无幻觉）"},
    "response_relevancy": {"label": "答案相关性", "needs_reference": False,
                           "desc": "答案是否切题（生成反推问题与原问题相似度）"},
    "context_recall": {"label": "上下文召回率", "needs_reference": True,
                       "desc": "参考答案是否都能被检索上下文覆盖"},
    "context_precision": {"label": "上下文精确率", "needs_reference": True,
                          "desc": "相关检索项是否排在前面（MRR 风格）"},
    # 0.1 别名兜底（列名可能不同）
    "answer_relevancy": {"label": "答案相关性", "needs_reference": False, "desc": "答案是否切题"},
}


def _is_nan(v) -> bool:
    """判断是否为 None / NaN（ragas 单样本异常时返回 NaN）。"""
    try:
        if v is None:
            return True
        if isinstance(v, float) and math.isnan(v):
            return True
        return False
    except Exception:
        return False


class RAGASEvaluationModule:
    """RAGAS 评估引擎：懒加载 LLM/embeddings，对样本集计算 RAGAS 指标。

    生命周期：
        构造（仅存配置，零成本） -> 首次 evaluate 时 _ensure_clients 加载模型 ->
        复用缓存的 LLM/embeddings 跑后续评估。

    Public API:
        - evaluate(samples, with_reference) -> {results, aggregates, count, metrics}
        - evaluate_single(question, answer, contexts, ground_truth=None) -> 同上（单样本）
        - load_eval_dataset() -> 内置测试集
    """

    # 内置测试集路径（相对本文件：backend/rag_modules/ -> ../data/eval_dataset.json）
    _DATASET_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "eval_dataset.json"
    )

    def __init__(self, config):
        """仅保存配置，不初始化任何模型（懒加载）。

        Args:
            config: GraphRAGConfig，提供 llm_model / embedding_model 等
        """
        self.config = config
        self._llm = None            # ChatOpenAI（DeepSeek 兼容）
        self._embeddings = None     # HuggingFaceEmbeddings（BGE）
        self._run_config = None     # RAGAS RunConfig
        self._dataset_cache: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # 懒加载
    # ------------------------------------------------------------------
    def _ensure_clients(self):
        """首次评估时初始化 LLM / embeddings / RunConfig（后续复用缓存）。"""
        if self._llm is not None:
            return
        _require_ragas()

        from langchain_openai import ChatOpenAI
        from langchain_huggingface import HuggingFaceEmbeddings

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("未设置 DEEPSEEK_API_KEY，无法初始化 RAGAS 评估 LLM")

        # judge 模型：优先环境变量，回退系统 LLM 模型
        judge_model = os.getenv("RAGAS_JUDGE_MODEL") or self.config.llm_model

        self._llm = ChatOpenAI(
            model=judge_model,
            base_url="https://api.deepseek.com",
            api_key=api_key,
            temperature=0,      # 评估需确定性，温度 0
            timeout=120,
        )
        # 复用 BGE embeddings（ResponseRelevancy 需要：生成反推问题并 embedding 比对）
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        max_workers = max(1, int(os.getenv("RAGAS_MAX_WORKERS", "3")))
        self._run_config = _RunConfig(
            max_workers=max_workers,
            max_retries=3,
            timeout=180,
        )
        logger.info(f"RAGAS 评估引擎就绪：judge={judge_model}, workers={max_workers}")

    # ------------------------------------------------------------------
    # 核心评估
    # ------------------------------------------------------------------
    def evaluate(
        self,
        samples: List[Dict[str, Any]],
        with_reference: bool,
    ) -> Dict[str, Any]:
        """对一批样本运行 RAGAS 评估。

        Args:
            samples: [{question, answer, contexts: [str], ground_truth?}]
            with_reference: 是否有 ground_truth（True 跑完整 4 指标，False 仅忠实度+相关性）

        Returns:
            {results: [{question, <metric>...}], aggregates: {metric: avg},
             count: int, metrics: [str]}
        """
        _require_ragas()
        if not samples:
            return {"results": [], "aggregates": {}, "count": 0, "metrics": []}

        self._ensure_clients()

        # 组装 RAGAS SingleTurnSample
        ragas_samples = []
        for s in samples:
            kwargs: Dict[str, Any] = {
                "user_input": s["question"],
                "response": s["answer"],
                "retrieved_contexts": s.get("contexts") or [],
            }
            if with_reference and s.get("ground_truth"):
                kwargs["reference"] = s["ground_truth"]
            ragas_samples.append(_SingleTurnSample(**kwargs))

        dataset = _EvaluationDataset(ragas_samples)

        # 按 reference 有无选指标
        metrics = [_Faithfulness(), _ResponseRelevancy()]
        if with_reference:
            metrics += [_LLMContextRecall(), _LLMContextPrecisionWithReference()]

        scorer = _evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=_LangchainLLMWrapper(self._llm),
            embeddings=_LangchainEmbeddingsWrapper(self._embeddings),
            run_config=self._run_config,
            raise_exceptions=False,   # 单样本异常返回 NaN，不中断整批
        )

        df = scorer.to_pandas()
        records = df.to_dict(orient="records")

        # 提取实际产出的指标列（兼容 0.1/0.2 列名差异）
        metric_keys = [k for k in METRIC_META if k in df.columns]
        results = []
        for i, rec in enumerate(records):
            item: Dict[str, Any] = {"question": samples[i].get("question", "")}
            for mk in metric_keys:
                val = rec.get(mk)
                item[mk] = None if _is_nan(val) else round(float(val), 4)
            results.append(item)

        aggregates: Dict[str, Any] = {}
        for mk in metric_keys:
            vals = [r[mk] for r in results if r[mk] is not None]
            aggregates[mk] = round(sum(vals) / len(vals), 4) if vals else None

        return {
            "results": results,
            "aggregates": aggregates,
            "count": len(results),
            "metrics": metric_keys,
        }

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """单样本评估（无 ground_truth 仅忠实度+相关性；有则完整 4 指标）。"""
        with_reference = bool(ground_truth)
        samples = [{
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        }]
        return self.evaluate(samples, with_reference=with_reference)

    # ------------------------------------------------------------------
    # 内置测试集
    # ------------------------------------------------------------------
    def load_eval_dataset(self) -> List[Dict[str, Any]]:
        """加载内置烹饪评估测试集（question + ground_truth，可缓存）。"""
        if self._dataset_cache is not None:
            return self._dataset_cache
        try:
            with open(self._DATASET_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._dataset_cache = data if isinstance(data, list) else []
        except FileNotFoundError:
            logger.warning(f"评估测试集不存在: {self._DATASET_PATH}")
            self._dataset_cache = []
        except Exception as e:
            logger.error(f"加载评估测试集失败: {e}")
            self._dataset_cache = []
        return self._dataset_cache
