"""
本地内存记忆实现

使用 Python 字典存储对话历史，无需外部依赖。
适合开发调试和小型应用。

注意：进程重启后数据丢失，生产环境请使用 RedisMemory。
"""
from collections import defaultdict

from memory.base import MemoryBase, MemoryMessage


class LocalMemory(MemoryBase):
    """
    本地内存记忆实现

    使用示例：
        memory = LocalMemory(max_messages=20)
        await memory.add("session_1", "user", "你好")
        await memory.add("session_1", "assistant", "你好！有什么可以帮你的？")
        history = await memory.get_history("session_1")
    """

    def __init__(self, max_messages: int = 100):
        """
        初始化本地记忆

        :param max_messages: 每个会话最大消息数
        """
        self._store: dict[str, list[MemoryMessage]] = defaultdict(list)
        self._max_messages = max_messages

    async def add(self, session_id: str, role: str, content: str) -> bool:
        """添加消息"""
        self._store[session_id].append(MemoryMessage(role=role, content=content))

        # 超出限制时裁剪
        if len(self._store[session_id]) > self._max_messages:
            self._store[session_id] = self._store[session_id][-self._max_messages:]

        return True

    async def get_history(
        self, session_id: str, limit: int = 10
    ) -> list[MemoryMessage]:
        """获取对话历史（滑动窗口）"""
        messages = self._store.get(session_id, [])
        return messages[-limit:]

    async def clear(self, session_id: str) -> bool:
        """清除会话历史"""
        self._store.pop(session_id, None)
        return True
