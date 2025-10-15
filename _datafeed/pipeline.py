# data/pipeline.py
import re
import asyncio
import pandas as pd
from typing import Dict, Any, List, Callable
from collections import defaultdict

from infra.ws_client import WSClient
from feature.handlers import channel_registry
from feature.integrate import SnapshotThrottler, build_snapshot_from_row

from utils.logger import logger

class DataPipeline:
    def __init__(self, 
                 cfg: dict, 
                 processor: None, 
                 max_queue = 100_000):
        self.cfg = cfg

        from feature.provider import build_ws_plan  # 新函数
        plan = build_ws_plan(cfg)

        df_auth = cfg.get("datafeed", {})
        auth = cfg.get("auth", {})
        api_key = df_auth.get("api_key", auth.get("api_key", ""))
        secret_key = df_auth.get("secret_key", auth.get("secret_key", ""))
        passphrase = df_auth.get("passphrase", auth.get("passphrase", ""))

        self.clients: List[WSClient] = []
        for item in plan:
            need_login = (item["ws_kind"] == "private") or bool(df_auth.get("need_login", False))
            self.clients.append(
                WSClient(
                    item["url"],
                    item["args"],
                    need_login=need_login,
                    api_key=api_key,
                    secret_key=secret_key,
                    passphrase=passphrase,
                    ping_interval=15,
                )
            )
        self.processor = processor
        self.q: asyncio.Queue[Dict] = asyncio.Queue(maxsize=max_queue)
        self._lock = asyncio.Lock()
        
    async def on_json(self, msg: Dict[str, Any]):
        if "event" in msg:
            return
        await self.q.put(msg)
        
    async def _consumer_loop(self):
        while True:
            msg = await self.q.get()
            try:
                await self.processor.handle(msg)
            except:
                logger.exception("processor failed")
            finally:
                self.q.task_done()

    async def run(self):
        ws_tasks = [asyncio.create_task(c.run_forever(self.on_json)) for c in self.clients]
        if not ws_tasks:
            return
        worker_n = max(1, int(self.cfg.get("engine", {}).get("workers", 1)))
        consume_tasks = [asyncio.create_task(self._consumer_loop()) for _ in range(worker_n)]
        try:
            await asyncio.gather(*ws_tasks)
        finally:
            await asyncio.gather(*[c.stop() for c in self.clients], return_exceptions=True)
            await self.q.join()
            for t in consume_tasks:
                t.cancel()
    
    async def stop(self):
        # await self.ws.stop()
        await asyncio.gather(*(c.stop() for c in self.clients))


class WriteProcessor:
    def __init__(self, cfg: dict, store):
        self.store = store
        self._handlers = channel_registry
        self._flush_n = cfg["datafeed"]["flush_every_n"]
        self._cnt = defaultdict(int)

    def __call__(self, msg: Dict[str, Any]):
        arg = msg.get("arg", {})
        ch = arg.get("channel", "")
        inst = arg.get("instId", "")
        
        for prefix, handler in self._handlers.items():
            if ch.startswith(prefix):
                df = handler(msg)
                if ch == "candle1m":
                    print(f"write {ch} {inst}, ts={msg.get('data', [0])[0]}")
                if df is None or df.empty:
                    return

                self.store.write(channel=ch, arg=arg, df=df)

                self._cnt[prefix] += len(df)
                flush_limit = int(self._flush_n.get(prefix, 9999999))
                if self._cnt[prefix] >= flush_limit:
                    self._cnt[prefix] = 0
                return
