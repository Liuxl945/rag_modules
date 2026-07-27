"""
聊天历史管理（多会话模型）：文件持久化（JSON）+ 内存缓存 + 线程锁

存储文件：backend/conversations.json，结构为一个会话列表，每个会话内嵌自己的消息：

    [
      {
        "id": "conv-1",
        "title": "红烧肉怎么做？",
        "created_at": 1785158194.233,
        "updated_at": 1785158355.909,
        "messages": [
          {"id": "msg-1", "role": "user", "content": "...", "analysis": null,
           "sources": null, "elapsed": null, "error": false, "timestamp": 1785158194.233}
        ]
      }
    ]

启动时若 conversations.json 不存在但旧的扁平 chat_history.json 存在且非空，
会把其全部消息折叠成一条「历史会话」做一次性迁移，旧文件保留不删。

接口（单例 `conversation_store`）：
- list_conversations()            -> list[dict]   仅元数据，按 updated_at 倒序
- get_conversation(conv_id)       -> dict | None  完整会话（含 messages）
- create_conversation(title=None) -> dict         新建空会话
- append_message(conv_id, ...)    -> dict | None  追加消息，返回 {"message", "conversation"}
- rename_conversation(conv_id, t) -> dict | None
- delete_conversation(conv_id)    -> bool
"""

import json
import os
import time
import threading
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONVERSATIONS_FILE = os.path.join(BASE_DIR, "conversations.json")
LEGACY_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

DEFAULT_TITLE = "新对话"
TITLE_MAX_LEN = 24

_lock = threading.Lock()


def _atomic_write(path: str, data: Any):
    """原子写：先写临时文件再 os.replace，避免并发/中断导致半截文件。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_conversations() -> List[Dict[str, Any]]:
    """从磁盘加载会话列表；文件不存在时返回空列表。"""
    if not os.path.exists(CONVERSATIONS_FILE):
        return []
    try:
        with open(CONVERSATIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def _migrate_from_legacy() -> Optional[List[Dict[str, Any]]]:
    """若旧扁平 chat_history.json 存在且非空，折叠成一条历史会话返回；否则 None。"""
    if not os.path.exists(LEGACY_HISTORY_FILE):
        return None
    try:
        with open(LEGACY_HISTORY_FILE, "r", encoding="utf-8") as f:
            msgs = json.load(f)
    except Exception:
        return None
    if not isinstance(msgs, list) or not msgs:
        return None

    # 标题取首条 user 消息，否则「历史对话」
    title = DEFAULT_TITLE
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
            title = str(m["content"])[:TITLE_MAX_LEN]
            break
    else:
        title = "历史对话"

    now = time.time()
    conv = {
        "id": "conv-1",
        "title": title,
        "created_at": msgs[0].get("timestamp", now) if isinstance(msgs[0], dict) else now,
        "updated_at": msgs[-1].get("timestamp", now) if isinstance(msgs[-1], dict) else now,
        "messages": msgs,
    }
    return [conv]


def _meta(conv: Dict[str, Any]) -> Dict[str, Any]:
    """从完整会话提取侧边栏需要的元数据（不含 messages 全文）。"""
    msgs = conv.get("messages") or []
    last_preview = ""
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("content"):
            last_preview = str(m["content"])[:60]
            break
    return {
        "id": conv["id"],
        "title": conv.get("title") or DEFAULT_TITLE,
        "created_at": conv.get("created_at"),
        "updated_at": conv.get("updated_at"),
        "message_count": len(msgs),
        "last_message_preview": last_preview,
    }


class ConversationStore:
    """多会话聊天历史存储（单进程内存缓存 + 文件持久化）。"""

    def __init__(self):
        self._conversations: List[Dict[str, Any]] = []
        self._conv_seq = 0
        self._msg_seq = 0
        self._init_store()

    def _init_store(self):
        """加载或迁移数据，并初始化 ID 序号。"""
        with _lock:
            convs = _load_conversations()
            if not convs:
                migrated = _migrate_from_legacy()
                if migrated is not None:
                    convs = migrated
                    try:
                        _atomic_write(CONVERSATIONS_FILE, convs)
                    except Exception:
                        pass
            self._conversations = convs
            self._init_seq()

    def _init_seq(self):
        """扫描已有数据，确定 conv-N / msg-N 的最大序号，避免 ID 冲突。"""
        max_conv = 0
        max_msg = 0
        for c in self._conversations:
            cid = c.get("id", "")
            if isinstance(cid, str) and cid.startswith("conv-"):
                try:
                    max_conv = max(max_conv, int(cid.split("-")[-1]))
                except ValueError:
                    pass
            for m in c.get("messages") or []:
                mid = m.get("id", "") if isinstance(m, dict) else ""
                if isinstance(mid, str) and mid.startswith("msg-"):
                    try:
                        max_msg = max(max_msg, int(mid.split("-")[-1]))
                    except ValueError:
                        pass
        self._conv_seq = max_conv
        self._msg_seq = max_msg

    def _next_conv_id(self) -> str:
        self._conv_seq += 1
        return f"conv-{self._conv_seq}"

    def _next_msg_id(self) -> str:
        self._msg_seq += 1
        return f"msg-{self._msg_seq}"

    def _persist(self):
        """落盘（调用方需持锁）。"""
        try:
            _atomic_write(CONVERSATIONS_FILE, self._conversations)
        except Exception:
            pass

    def _find(self, conv_id: str) -> Optional[Dict[str, Any]]:
        for c in self._conversations:
            if c.get("id") == conv_id:
                return c
        return None

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def list_conversations(self) -> List[Dict[str, Any]]:
        """返回会话元数据列表，按 updated_at 倒序（最近在前）。"""
        with _lock:
            metas = [_meta(c) for c in self._conversations]
        metas.sort(key=lambda m: m.get("updated_at") or 0, reverse=True)
        return metas

    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        with _lock:
            c = self._find(conv_id)
            return json.loads(json.dumps(c)) if c else None  # 深拷贝，避免外部误改缓存

    def create_conversation(self, title: Optional[str] = None) -> Dict[str, Any]:
        now = time.time()
        with _lock:
            conv = {
                "id": self._next_conv_id(),
                "title": (title or DEFAULT_TITLE),
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }
            self._conversations.append(conv)
            self._persist()
            return json.loads(json.dumps(conv))

    def append_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        analysis: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        elapsed: Optional[float] = None,
        error: Optional[bool] = False,
        timestamp: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """向指定会话追加一条消息。

        若会话标题仍是默认占位且本次是首条 user 消息，自动用其内容前缀作为标题。
        返回 {"message": msg, "conversation": meta}；会话不存在返回 None。
        """
        with _lock:
            conv = self._find(conv_id)
            if conv is None:
                return None
            msg = {
                "id": self._next_msg_id(),
                "role": role,
                "content": content,
                "analysis": analysis,
                "sources": sources,
                "elapsed": elapsed,
                "error": bool(error),
                "timestamp": timestamp if timestamp is not None else time.time(),
            }
            conv.setdefault("messages", []).append(msg)
            conv["updated_at"] = msg["timestamp"]

            # 首条 user 消息且标题仍为默认占位 -> 自动取标题
            if (
                role == "user"
                and content
                and (not conv.get("title") or conv.get("title") == DEFAULT_TITLE)
            ):
                conv["title"] = content[:TITLE_MAX_LEN]

            self._persist()
            return {"message": msg, "conversation": _meta(conv)}

    def rename_conversation(self, conv_id: str, title: str) -> Optional[Dict[str, Any]]:
        with _lock:
            conv = self._find(conv_id)
            if conv is None:
                return None
            conv["title"] = (title or DEFAULT_TITLE)[:TITLE_MAX_LEN] or DEFAULT_TITLE
            self._persist()
            return _meta(conv)

    def delete_conversation(self, conv_id: str) -> bool:
        with _lock:
            for i, c in enumerate(self._conversations):
                if c.get("id") == conv_id:
                    del self._conversations[i]
                    self._persist()
                    return True
            return False


# 单例
conversation_store = ConversationStore()
