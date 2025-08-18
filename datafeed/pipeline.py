# data/pipeline.py
import re
import asyncio
import pandas as pd
from typing import Dict, Any, List, Callable
from collections import defaultdict

from datafeed.websockets import WSClient
from datafeed.handlers import channel_registry
from datafeed.storage import CompositeStore


class DataPipeline:
    def __init__(self, cfg: dict, processor: Callable[[dict], None]):
        self.cfg = cfg

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
        self.processor = processor
        
    async def on_json(self, 
                      msg: Dict[str, Any], 
                      ):
        
        if "event" in msg:
            return
        if self.processor is not None:
            self.processor(msg)

    async def run(self):
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


class WriteProcessor:
    def __init__(self, cfg: dict, store: CompositeStore):
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
            
class FeatureEngineProcessor:
    _tf_pat = re.compile(r"(?:candle)(\d+(?:ms|s|m|h|d))")
    def __init__(self, cfg: dict, engine: Any):
        self.engine = engine

    def __call__(self, msg: Dict[str, Any]):
        """
        输入：原始 WS 消息（含 arg/data），用你现有的 handler 解析后喂引擎
        返回：当且仅当是“bar 收盘”产生特征时，返回 features DataFrame；否则返回 None
        """
        arg = msg.get("arg", {})
        channel = arg.get("channel","")
        instId  = arg.get("instId","")
        prefix = None
        for k in channel_registry.keys():
            if channel.startswith(k):
                prefix = k; break
        if not prefix:
            return None

        df = channel_registry[prefix](msg)
        if df is None or df.empty:
            return None

        if prefix in ("candle", "mark-price-candle", "index-candle"):
            tf = self._extract_tf(channel) or "1m"
            feats = self.engine.update_candles(df, instId=instId, tf=tf)
            # print(feats)
            print(prefix, tf, len(feats))
            return feats if not feats.empty else None
        elif prefix == "books":
            self.engine.update_books(df, instId=instId, tf="1m")
            return None
        elif prefix == "trades":
            self.engine.update_trades(df, instId=instId, tf="1m")
            return None
        else:
            return None

    def _extract_tf(self, channel: str):
        m = self._tf_pat.search(channel)
        return m.group(1) if m else None