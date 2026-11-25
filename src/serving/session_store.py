"""Bounded Redis sliding-window session store (LPUSH + LTRIM 0 19 + TTL)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass(frozen=True)
class SessionEvent:
    """Compact click/interaction record stored in Redis."""

    article_id: str
    price: float
    department_name: str
    product_group_name: str


class RedisLike(Protocol):
    """Minimal Redis interface for dependency injection and testing."""

    def pipeline(self): ...
    def lrange(self, name: str, start: int, end: int) -> list: ...


class InMemoryRedis:
    """Lightweight in-memory Redis drop-in for unit tests and offline dev."""

    def __init__(self) -> None:
        self._lists: dict[str, list[str]] = {}
        self._ttls: dict[str, int] = {}

    def pipeline(self):
        return _InMemoryPipeline(self)

    def lpush(self, key: str, value: str) -> int:
        buf = self._lists.setdefault(key, [])
        buf.insert(0, value)
        return len(buf)

    def ltrim(self, key: str, start: int, end: int) -> bool:
        buf = self._lists.get(key, [])
        self._lists[key] = buf[start : end + 1]
        return True

    def expire(self, key: str, seconds: int) -> bool:
        self._ttls[key] = seconds
        return True

    def lrange(self, key: str, start: int, end: int) -> list[str]:
        buf = self._lists.get(key, [])
        return buf[start : end + 1] if end >= 0 else buf[start:]

    def ping(self) -> bool:
        return True


class _InMemoryPipeline:
    def __init__(self, store: InMemoryRedis) -> None:
        self._store = store
        self._ops: list = []

    def lpush(self, key: str, value: str):
        self._ops.append((self._store.lpush, (key, value)))
        return self

    def ltrim(self, key: str, start: int, end: int):
        self._ops.append((self._store.ltrim, (key, start, end)))
        return self

    def expire(self, key: str, seconds: int):
        self._ops.append((self._store.expire, (key, seconds)))
        return self

    def execute(self) -> list:
        return [fn(*args) for fn, args in self._ops]


class SessionStore:
    """Manages bounded 20-item sliding windows per session in Redis."""

    def __init__(self, redis_client: RedisLike, max_len: int = 20, ttl_seconds: int = 86400) -> None:
        self._redis = redis_client
        self._max_len = max_len
        self._ttl = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}:history"

    def record_event(self, session_id: str, event: SessionEvent) -> int:
        """Push newest event to head of list, trim to max_len, refresh TTL."""
        key = self._key(session_id)
        payload = json.dumps(asdict(event))
        pipe = self._redis.pipeline()
        pipe.lpush(key, payload)
        pipe.ltrim(key, 0, self._max_len - 1)
        pipe.expire(key, self._ttl)
        results = pipe.execute()
        return min(int(results[0]), self._max_len)

    def get_history(self, session_id: str) -> list[SessionEvent]:
        """Return up to max_len most-recent events (index 0 = latest)."""
        raw_items = self._redis.lrange(self._key(session_id), 0, self._max_len - 1)
        events: list[SessionEvent] = []
        for raw in raw_items:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(raw)
            events.append(SessionEvent(**data))
        return events
