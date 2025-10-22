# feature/manager.py
import asyncio, contextlib
from typing import Dict, Any, List

from feature.inst_worker import InstrumentWorker
from utils.logger import logger


class WorkerManager:
    def __init__(self, cfg: Dict[str, Any], redis_dsn: str, stream_name: str = "features"):
        self.cfg = cfg
        self.redis_dsn = redis_dsn
        self.stream_name = stream_name
        self.workers: Dict[str, InstrumentWorker] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    async def start_from_cfg(self):
        insts = self.cfg.get("datafeed", {}).get("instIds", []) or []
        for inst in insts:
            await self.add_instrument(inst)

    async def stop_all(self):
        async with self._lock:
            insts = list(self.workers.keys())
        for inst in insts:
            await self.remove_instrument(inst)
            await asyncio.sleep(0.1)

    async def add_instrument(self, inst: str):
        async with self._lock:
            if inst in self.workers:
                return
            cleaned_cfg = self.cfg
            cleaned_cfg["datafeed"]["instIds"] = [inst]
            w = InstrumentWorker(inst, cleaned_cfg, self.redis_dsn, self.stream_name)
            self.workers[inst] = w
        await w.start()

    async def remove_instrument(self, inst: str):
        async with self._lock:
            t = self.tasks.pop(inst, None)
            w = self.workers.pop(inst, None)
        if w:
            await w.stop()
        if t:
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await t

    async def apply_delta_to(self, inst: str, new_cfg_fragment: Dict[str, Any]):
        async with self._lock:
            w = self.workers.get(inst)
        if not w:
            await self.add_instrument(inst)
            async with self._lock:
                w = self.workers[inst]
        await w.apply_delta(new_cfg_fragment)

    async def list_instruments(self) -> List[str]:
        async with self._lock:
            return list(self.workers.keys())

    async def status(self) -> Dict[str, Any]:
        async with self._lock:
            data = {}
            for inst, w in self.workers.items():
                data[inst] = {
                    "ws_clients": list(w.clients.keys()),  # ["public"] / ["business"] / ["public","business"]
                    "desired": {k: sorted(list(v)) for k, v in w._desired.items()},
                    "queue_size": w._q.qsize(),
                }
            return data