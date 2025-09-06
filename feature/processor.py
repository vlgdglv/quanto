# feature/processor.py
import re
import asyncio
import pandas as pd
from typing import Dict, Any, List, Callable
from collections import defaultdict

from infra.ws_client import WSClient
from datafeed.handlers import channel_registry
from datafeed.storage import CompositeStore
from feature.integrate import SnapshotThrottler, build_snapshot_from_row

from utils.logger import logger

class FeatureEngineProcessor:
    _tf_pat = re.compile(r"(?:candle)(\d+(?:m|H|D|M|Mutc|Wutc|Dutc|Hutc))")
    def __init__(self, cfg: dict, engine: Any, 
                 on_snapshot: Callable[[dict], None] = None,
                 feature_writer=None):
        self.engine = engine

        agent_cfg = ()
        agent_cfg = (cfg or {}).get("agent", {})
        min_int_ms = int(agent_cfg.get("min_snapshot_interval_ms", 300_000))
        align_5m   = bool(agent_cfg.get("align_to_5m", False))
        self.throttler = SnapshotThrottler(min_interval_ms=min_int_ms, align_to_5m=align_5m)
        self.on_snapshot = on_snapshot

        self.feature_writer = feature_writer
        self._tick_tf = str(agent_cfg.get("tick_tf", "1m"))
        self._update_map = {
            "candle": ("update_candles", True),
            "mark-price-candle": ("update_candles", True),
            "index-candle": ("update_candles", True),
            "books": ("update_books", False),
            "trades": ("update_trades", False),
            "funding-rate": ("update_funding_rate", False),
            "open-interest": ("update_open_interest", False),
        }

    def __call__(self, msg: Dict[str, Any]):
        """
        输入：原始 WS 消息（含 arg/data），用你现有的 handler 解析后喂引擎
        返回：当且仅当是“bar 收盘”产生特征时，返回 features DataFrame；否则返回 None
        """
        # logger.info("Processor called.")
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

        method_name, may_return_feats = self._update_map.get(prefix, (None, False))
        if not method_name:
            return None
        
        # 决定 tf：candle 从频道解析，其它挂到 _tick_tf
        if may_return_feats:
            tf = self._extract_tf(channel) or "1m"
            feats = getattr(self.engine, method_name)(df, instId=instId, tf=tf)
        else:
            tf = self._tick_tf
            getattr(self.engine, method_name)(df, instId=instId, tf=tf)
            feats = None
        
        # 仅当引擎返回了特征行（通常是 bar 收盘 confirm==1）才快照/写入
        if feats is not None and not feats.empty:
            rows = self.throttler.ingest_features_df(feats, policy="bar_close")
            if rows and self.on_snapshot:
                for row in rows:
                    snapshot = build_snapshot_from_row(row)
                    self.on_snapshot(snapshot)
            if self.feature_writer is not None:
                updates_msg = f"InstId: {instId}, tf: {tf}, total {self.engine.updates_cnt} updates in this slot."
                logger.info(updates_msg)
                self.feature_writer.add(feats)
            return feats if not feats.empty else None
        return None

    def _extract_tf(self, channel: str):
        m = self._tf_pat.search(channel)
        return m.group(1) if m else None