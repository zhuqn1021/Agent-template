"""
Redis 记忆实现

使用 Redis List 存储对话历史，支持：
- 滑动窗口
- TTL 自动过期
- 多进程/多实例共享

生产环境推荐使用。
"""
import json
import logging
from typing import Optional

from memory.base import MemoryBase, MemoryMessage

logger = logging.getLogger(__name__)

HISTORY_KEY_PREFIX = "agent:history"
DEFAULT_TTL = 86400  # 24 小时


class RedisMemory(MemoryBase):
    """
    Redis 记忆实现

    使用示例：
        import redis.asyncio as redis
        client = redis.Redis(host="localhost", port=6379)
        memory = RedisMemory(client, ttl=3600)

        await memory.add("session_1", "user", "你好")
        history = await memory.get_history("session_1", limit=10)
    """

    def __init__(self, redis_client, ttl: int = DEFAULT_TTL, key_prefix: str = HISTORY_KEY_PREFIX):
        """
        初始化 Redis 记忆

        :param redis_client: Redis 异步客户端
        :param ttl: 记忆过期时间（秒）
        :param key_prefix: Redis key 前缀
        """
        self._client = redis_client
        self._ttl = ttl
        self._key_prefix = key_prefix

    def _key(self, session_id: str) -> str:
        """生成 Redis key"""
        return f"{self._key_prefix}:{session_id}"

    async def add(self, session_id: str, role: str, content: str) -> bool:
        """添加消息到 Redis List"""
        try:
            key = self._key(session_id)
            message = json.dumps({"role": role, "content": content}, ensure_ascii=False)
            await self._client.rpush(key, message)
            await self._client.expire(key, self._ttl)
            return True
        except Exception as e:
            logger.error(f"Redis 记忆写入失败: {e}")
            return False

    async def get_history(
        self, session_id: str, limit: int = 10
    ) -> list[MemoryMessage]:
        """获取对话历史（滑动窗口）"""
        try:
            key = self._key(session_id)
            raw_messages = await self._client.lrange(key, -limit, -1)
            messages = []
            for msg in raw_messages:
                try:
                    data = json.loads(msg)
                    messages.append(MemoryMessage(role=data["role"], content=data["content"]))
                except (json.JSONDecodeError, KeyError):
                    continue
            return messages
        except Exception as e:
            logger.error(f"Redis 记忆读取失败: {e}")
            return []

    async def clear(self, session_id: str) -> bool:
        """清除会话历史"""
        try:
            key = self._key(session_id)
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis 记忆清除失败: {e}")
            return False
