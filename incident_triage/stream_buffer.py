from __future__ import annotations

import os
from collections import deque
from threading import Lock
from typing import Dict, List, Optional, Protocol, runtime_checkable

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover
    redis = None


@runtime_checkable
class StreamBufferProtocol(Protocol):
    def record_and_get_window(self, entity_id: str, value: float) -> List[float]: ...
    def get_window(self, entity_id: str) -> List[float]: ...
    def clear(self, entity_id: Optional[str] = None) -> None: ...


class InMemoryStreamingBuffer:
    """Thread-safe sliding-window buffer for single-node / local environments."""

    def __init__(self, maxlen: int = 30) -> None:
        self.maxlen = maxlen
        self._buffers: Dict[str, deque[float]] = {}
        self._lock = Lock()

    def record_and_get_window(self, entity_id: str, value: float) -> List[float]:
        with self._lock:
            if entity_id not in self._buffers:
                self._buffers[entity_id] = deque(maxlen=self.maxlen)
            self._buffers[entity_id].append(float(value))
            return list(self._buffers[entity_id])

    def get_window(self, entity_id: str) -> List[float]:
        with self._lock:
            return list(self._buffers.get(entity_id, []))

    def clear(self, entity_id: Optional[str] = None) -> None:
        with self._lock:
            if entity_id:
                self._buffers.pop(entity_id, None)
            else:
                self._buffers.clear()


class RedisStreamingBuffer:
    """Distributed atomic sliding-window buffer using Redis LPUSH + LTRIM.
    
    Enables multi-worker Uvicorn and multi-pod Kubernetes instances to share rolling state.
    """

    def __init__(self, redis_url: str, maxlen: int = 30) -> None:
        if redis is None:  # pragma: no cover
            raise RuntimeError("redis package is not installed.")
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.maxlen = maxlen

    def record_and_get_window(self, entity_id: str, value: float) -> List[float]:
        key = f"telemetry:{entity_id}"
        pipe = self.client.pipeline()
        pipe.lpush(key, str(value))
        pipe.ltrim(key, 0, self.maxlen - 1)
        pipe.lrange(key, 0, self.maxlen - 1)
        _, _, raw_values = pipe.execute()
        # raw_values are ordered from newest to oldest; reverse so oldest is first
        return [float(x) for x in reversed(raw_values)]

    def get_window(self, entity_id: str) -> List[float]:
        key = f"telemetry:{entity_id}"
        raw_values = self.client.lrange(key, 0, self.maxlen - 1)
        return [float(x) for x in reversed(raw_values)]

    def clear(self, entity_id: Optional[str] = None) -> None:
        if entity_id:
            self.client.delete(f"telemetry:{entity_id}")
        else:
            keys = self.client.keys("telemetry:*")
            if keys:
                self.client.delete(*keys)


def create_stream_buffer(maxlen: int = 30) -> StreamBufferProtocol:
    """Factory that creates a distributed Redis buffer if REDIS_URL is set, else in-memory."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url and redis is not None:
        try:
            buf = RedisStreamingBuffer(redis_url=redis_url, maxlen=maxlen)
            buf.client.ping()
            return buf
        except Exception:  # pragma: no cover
            pass
    return InMemoryStreamingBuffer(maxlen=maxlen)


# Backward-compatible alias for existing imports
StreamingWindowBuffer = InMemoryStreamingBuffer
