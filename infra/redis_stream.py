# infra/redis.py
import json
import asyncio, time, json, contextlib
import redis.asyncio as aioredis
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Mapping, Sequence

from utils.logger import logger

class RedisStreamsPublisher:
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
        logger.info(f"Redis stream {self._stream} initialized.")

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


FeatureCallback = Callable[[dict], Awaitable[None]]
try:
    import orjson as _json
    def _loads(b: bytes) -> Any: return _json.loads(b)
except Exception:
    def _loads(b: bytes) -> Any: return json.loads(b)


class RedisStreamsSubscriber:
    def __init__(
        self,
        dsn: str,
        stream: str = "features",
        *,
        start: str = "now",
        block_ms: int = 5000,
        fetch_count: int = 100,
        concurrency: int = 1,
        decode_responses: bool = False
    ):
        self._dsn = dsn
        self._stream = stream
        self._block_ms = block_ms
        self._fetch_count = fetch_count
        self._concurrency = max(1, concurrency)
        self._decode_responses = decode_responses
        self._r: Optional[aioredis.Redis] = None
        self._stop = False

        if start == "now":
            self._last_id: bytes = b"$"
        elif start == "earliest":
            self._last_id = b"0-0"
        else:
            self._last_id = start.encode() if isinstance(start, str) else start

        self._sem = asyncio.Semaphore(self._concurrency)

    async def _conn(self) -> aioredis.Redis:
        if self._r is None:
            self._r = await aioredis.from_url(self._dsn, decode_responses=False)
        return self._r

    async def run_forever(self, on_message: FeatureCallback):
        r = await self._conn()
        while not self._stop:
            try:
                resp = await r.xread({self._stream: self._last_id}, block=self._block_ms, count=self._fetch_count)
                if not resp:
                    continue
                # resp: [(b'stream', [(b'169...-0', {b'data': b'...json...'}), ...])]
                _, entries = resp[0]
                for entry_id, fields in entries:
                    raw = fields.get(b"data")
                    if not raw:
                        self._last_id = entry_id
                        continue
                    try:
                        payload = json.loads(raw)
                    except Exception:
                        logger.warning(f"[subscriber] json decode error id={entry_id!r}")
                        self._last_id = entry_id
                        continue

                    # 轻并发：并发执行回调，避免慢处理阻塞拉取
                    await self._sem.acquire()
                    asyncio.create_task(self._run_cb(on_message, payload, entry_id))
                    self._last_id = entry_id
                    
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[subscriber] loop error: {repr(e)}")
                await asyncio.sleep(0.5)

    async def _run_cb(self, cb: FeatureCallback, payload: dict, entry_id: bytes):
        try:
            await cb(payload)
        except Exception as e:
            logger.warning(f"[subscriber] callback error for id={entry_id!r}: {repr(e)}")
        finally:
            self._sem.release()

    async def stop(self):
        self._stop = True