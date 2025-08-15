# data/pipeline.py
import asyncio
import pandas as pd
from typing import Dict, Any, List
from collections import defaultdict

from datafeed.websockets import WSClient
from datafeed.handlers import channel_registry
from datafeed.storage import CompositeStore


class DataPipeline:
    def __init__(self, cfg: dict, store: CompositeStore):
        self.cfg = cfg
        self.store = store

        from datafeed.provider import build_ws_plan  # 新函数
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
                    ping_interval=20,
                )
            )
        
        self._handlers = channel_registry
        self._flush_n = cfg["datafeed"]["flush_every_n"]
        self._cnt = defaultdict(int)
        
    async def on_json(self, msg: Dict[str, Any]):
        arg = msg.get("arg", {})
        ch = arg.get("channel", "")
        inst = arg.get("instId", "")
        
        for prefix, handler in self._handlers.items():
            if ch.startswith(prefix):
                df = handler(msg)
                if df is None or df.empty:
                    return
                
                # df["instId"] = inst
                # self.store.write_df(prefix, df)
                if ch.startswith(("candle", "mark-price-candle", "index-candle")):
                    bar = ch.replace("mark-price-candle","").replace("index-candle","").replace("candle","")
                    self.store.write_candle(inst, prefix, bar, df)
                elif prefix == "trades":
                    self.store.write_trades(df)
                elif prefix == "books":
                    self.store.write_books(df)
                elif prefix == "funding-rate":
                    self.store.write_funding_rate(df)
                else:
                    return

                self._cnt[prefix] += len(df)
                flush_limit = int(self._flush_n.get(prefix, 9999999))
                if self._cnt[prefix] >= flush_limit:
                    self._cnt[prefix] = 0
                return

    async def run(self):
        # await self.ws.run_forever(self.on_json)
        tasks = [asyncio.create_task(c.run_forever(self.on_json)) for c in self.clients]
        if not tasks:
            return 
        try:
            await asyncio.gather(*tasks)
        finally:
            await asyncio.gather(*[c.stop() for c in self.clients])
    
    async def stop(self):
        # await self.ws.stop()
        await asyncio.gather(*(c.stop() for c in self.clients))