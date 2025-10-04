# infra/publisher.py
import json
import asyncio
from typing import Mapping, Any, Optional
# import aioredis
from redis.asyncio import Redis

class FeatureBusPublisher:
    async def publish(self, stream: str, payload: Mapping[str, Any]) -> None:
        raise NotImplementedError

class RedisStreamsPublisher(FeatureBusPublisher):
    def __init__(self, 
                 dsn: str, 
                 stream: str = "features", 
                 maxlen_approx: Optional[int] = 1_000_000
                 ):
        self._dsn = dsn
        self._stream = stream
        self._maxlen = maxlen_approx
        self._redis: Optional[aioredis.Redis] = None
        self._lock = asyncio.Lock()

    async def _conn(self):
        if self._redis is None:
            async with self._lock:
                if self._redis is None:
                    self._redis = await aioredis.from_url(self._dsn, decode_responses=False)
        return self._redis

    async def publish(self, stream: Optional[str], payload: Mapping[str, Any]) -> None:
        r = await self._conn()
        stream = stream or self._stream
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        await r.xadd(stream, {"data": data}, maxlen=self._maxlen, approximate=True)
