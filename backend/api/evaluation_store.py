"""
RAGAS 评估结果持久化（单例）：文件持久化（JSON）+ 内存缓存 + 线程锁

存储文件：backend/evaluation_results.json，结构为评估运行列表：

    [
      {
        "id": "eval-1",
        "created_at": 1785158194.233,
        "kind": "dataset" | "single",
        "count": 8,
        "metrics": ["faithfulness", ...],
        "aggregates": {"faithfulness": 0.82, ...},
        "results": [...],
        "elapsed": 45.2
      }
    ]

接口（单例 `evaluation_store`）：
- list_runs()          -> list[dict]   仅元数据，按 created_at 倒序
- get_run(run_id)      -> dict | None  完整运行
- save_run(run)        -> dict         保存并返回元数据
- delete_run(run_id)   -> bool
"""

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_FILE = os.path.join(BASE_DIR, "evaluation_results.json")

_lock = threading.Lock()


def _atomic_write(path: str, data: Any):
    """原子写：先写临时文件再 os.replace，避免并发/中断导致半截文件。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_runs() -> List[Dict[str, Any]]:
    """从磁盘加载评估运行列表；文件不存在时返回空列表。"""
    if not os.path.exists(RESULTS_FILE):
        return []
    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


class EvaluationStore:
    """RAGAS 评估结果存储（单进程内存缓存 + 文件持久化）。"""

    def __init__(self):
        self._runs: List[Dict[str, Any]] = []
        self._seq = 0
        self._init_store()

    def _init_store(self):
        """加载数据并初始化 ID 序号（避免 ID 冲突）。"""
        with _lock:
            self._runs = _load_runs()
            max_seq = 0
            for r in self._runs:
                rid = r.get("id", "") if isinstance(r, dict) else ""
                if isinstance(rid, str) and rid.startswith("eval-"):
                    try:
                        max_seq = max(max_seq, int(rid.split("-")[-1]))
                    except ValueError:
                        pass
            self._seq = max_seq

    def _next_id(self) -> str:
        self._seq += 1
        return f"eval-{self._seq}"

    def _persist(self):
        """落盘（调用方需持锁）。"""
        try:
            _atomic_write(RESULTS_FILE, self._runs)
        except Exception:
            pass

    @staticmethod
    def _meta(run: Dict[str, Any]) -> Dict[str, Any]:
        """从完整运行提取列表所需元数据（不含 results 全文）。"""
        return {
            "id": run.get("id"),
            "created_at": run.get("created_at"),
            "kind": run.get("kind"),
            "count": run.get("count", 0),
            "metrics": run.get("metrics", []),
            "aggregates": run.get("aggregates", {}),
            "elapsed": run.get("elapsed"),
        }

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def list_runs(self) -> List[Dict[str, Any]]:
        """返回评估运行元数据列表，按 created_at 倒序（最近在前）。"""
        with _lock:
            metas = [self._meta(r) for r in self._runs]
        metas.sort(key=lambda m: m.get("created_at") or 0, reverse=True)
        return metas

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with _lock:
            for r in self._runs:
                if r.get("id") == run_id:
                    return json.loads(json.dumps(r))  # 深拷贝，避免外部误改缓存
        return None

    def save_run(
        self,
        kind: str,
        results: List[Dict[str, Any]],
        aggregates: Dict[str, Any],
        metrics: List[str],
        elapsed: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """保存一次评估运行，返回完整运行（含 id / created_at）。

        Args:
            kind: "dataset" | "single"
            results: 每样本指标得分列表
            aggregates: 各指标聚合均值
            metrics: 指标列名列表
            elapsed: 总耗时（秒）
            extra: 附加信息（如 skipped / question 预览等）
        """
        now = time.time()
        with _lock:
            run: Dict[str, Any] = {
                "id": self._next_id(),
                "created_at": now,
                "kind": kind,
                "count": len(results),
                "metrics": metrics,
                "aggregates": aggregates,
                "results": results,
                "elapsed": elapsed,
            }
            if extra:
                run["extra"] = extra
            self._runs.append(run)
            self._persist()
            return json.loads(json.dumps(run))

    def delete_run(self, run_id: str) -> bool:
        with _lock:
            for i, r in enumerate(self._runs):
                if r.get("id") == run_id:
                    del self._runs[i]
                    self._persist()
                    return True
            return False


# 单例
evaluation_store = EvaluationStore()
