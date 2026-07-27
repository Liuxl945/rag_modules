"""
全局应用状态 - 持有 RAG 系统实例与就绪状态

独立成模块以避免 app.py 与 routes.py 之间的循环导入：
    app.py   -> state（lifespan 中写入 system / status）
    routes.py -> state（请求时读取 system / status）
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AppState:
    """单例式应用状态：RAG 系统实例 + 初始化状态。

    status 取值：
        - initializing: 后台初始化中
        - ready:        系统就绪，可接受查询
        - error:        初始化失败（error 字段存原因）
    """

    def __init__(self):
        self.system = None  # AdvancedGraphRAGSystem 实例（初始化完成后赋值）
        self.status = "initializing"
        self.error: Optional[str] = None

    @property
    def ready(self) -> bool:
        """系统是否就绪：status=ready 且 system 已就绪。"""
        return (
            self.status == "ready"
            and self.system is not None
            and getattr(self.system, "system_ready", False)
        )


# 全局共享实例
state = AppState()
