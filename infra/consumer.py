import asyncio, json, logging, signal
from typing import Callable, Awaitable, Optional, Dict, Any, Tuple, List
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

class StreamConsumer:
    def __init__(
        self,
        dsn: str,
        stream: str,
        group: str,
        consumer_name: str,
        handler: Callable[[Dict[str, Any]], Awaitable[None]],
        block_ms: int = 5000,
        count: int = 256,
        claim_idle_ms: int = 60_000,
        claim_batch: int = 512,
    ):
        self.dsn = dsn
        self.stream = stream
        self.group = group
        self.consumer_name = consumer_name
        self.handler = handler
        self.block_ms = block_ms
        self.count = count
        self.claim_idle_ms = claim_idle_ms
        self.claim_batch = claim_batch
        self._r: Optional[aioredis.Redis] = None
        self._stop = asyncio.Event()

    async def _conn(self):
        if self._r is None:
            self._r = await aioredis.from_url(self.dsn, decode_responses=False)
        return self._r

    async def ensure_group(self):
        r = await self._conn()
        try:
            await r.xgroup_create(self.stream, self.group, id="$", mkstream=True)
            logger.info("Created group %s on stream %s", self.group, self.stream)
        except aioredis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info("Group %s already exists", self.group)
            else:
                raise

    async def _read_once(self) -> List[Tuple[bytes, List[Tuple[bytes, Dict[bytes, bytes]]]]]:
        r = await self._conn()
        return await r.xreadgroup(
            groupname=self.group,
            consumername=self.consumer_name,
            streams={self.stream: ">"},
            count=self.count,
            block=self.block_ms,
        )

    async def _ack(self, entry_id: bytes):
        r = await self._conn()
        await r.xack(self.stream, self.group, entry_id)

    async def _auto_claim(self):
        r = await self._conn()
        start_id = b"0-0"
        claimed_any = False
        while True:
            res = await r.xautoclaim(self.stream, self.group, self.consumer_name,
                                     min_idle_time=self.claim_idle_ms, start=start_id, count=self.claim_batch)
            # res: (next_start_id, [(entry_id, {b'data': b'...'})])
            next_start_id, entries = res
            if not entries:
                break
            claimed_any = True
            for entry_id, fields in entries:
                try:
                    data = json.loads(fields[b"data"])
                    await self.handler(data)
                    await self._ack(entry_id)
                except Exception:
                    logger.exception("Handler failed for claimed entry %s", entry_id)
            if next_start_id == start_id:
                break
            start_id = next_start_id
        if claimed_any:
            logger.info("Auto-claimed some pending messages for group=%s", self.group)

    async def run(self):
        await self.ensure_group()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._stop.set)

        claim_task = asyncio.create_task(self._claim_loop())

        try:
            while not self._stop.is_set():
                resp = await self._read_once()
                if not resp:
                    continue
                # resp: [(b'features', [(b'168...-0', {b'data': b'...'}), ...])]
                for _, entries in resp:
                    for entry_id, fields in entries:
                        try:
                            data = json.loads(fields[b"data"])
                            await self.handler(data)
                            await self._ack(entry_id)
                        except Exception:
                            logger.exception("Handler failed for entry %s", entry_id)
        finally:
            claim_task.cancel()
            with asyncio.CancelledError:
                pass

    async def _claim_loop(self):
        while not self._stop.is_set():
            try:
                await self._auto_claim()
            except Exception:
                logger.exception("auto-claim loop error")
            await asyncio.sleep(max(self.claim_idle_ms // 1000, 10))