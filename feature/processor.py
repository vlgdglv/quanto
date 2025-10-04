# feature/processor.py
import re
import asyncio
import pandas as pd
from typing import Dict, Any, Optional, Callable, Awaitable, Tuple

from collections import defaultdict

from infra.ws_client import WSClient
from infra.publisher import FeatureBusPublisher
from datafeed.handlers import channel_registry
from datafeed.storage import CompositeStore
from feature.integrate import SnapshotThrottler, build_snapshot_from_row

from utils.logger import logger

class FeatureEngineProcessor:
    _tf_pat = re.compile(r"(?:candle)(\d+(?:m|H|D|M|Mutc|Wutc|Dutc|Hutc))")
    def __init__(self, 
                 cfg: dict, engine: Any, 
                 on_snapshot: Callable[[dict], None] = None,
                 feature_writer=None,
                 publisher=None,
                 stream_name: str = "features"
                 ):
        self.engine = engine

        agent_cfg = ()
        agent_cfg = (cfg or {}).get("agent", {})
        # min_int_ms = int(agent_cfg.get("min_snapshot_interval_ms", 300_000))
        # align_5m   = bool(agent_cfg.get("align_to_5m", False))
        # self.throttler = SnapshotThrottler(min_interval_ms=min_int_ms, align_to_5m=align_5m)
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

        self.on_rows_async: Optional[Callable[[list[dict]], Awaitable[None]]] = None
        self.on_rows_async = on_snapshot
        self._emit_last_ts: Dict[Tuple[str, str], Optional[int]] = {}
        self.publisher = publisher
        self.stream_name = stream_name
        

    async def handle(self, msg: Dict[str, Any]) -> None:
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
        
        if may_return_feats:
            tf = self._extract_tf(channel) or "1m"
            feats = getattr(self.engine, method_name)(df, instId=instId, tf=tf)
        else:
            tf = self._tick_tf
            getattr(self.engine, method_name)(df, instId=instId, tf=tf)
            feats = None
        
        if feats is None or feats.empty:
            return

        rows = self._collect_bar_close_rows(feats, instId=instId, tf=tf)
        if not rows:
            return

        if self.on_rows_async:
            # asyncio.create_task(self.on_rows_async(build_snapshot_from_row(rows)))
            for row in rows:
                snap = build_snapshot_from_row(row)
                self.on_snapshot(snap)
        elif self.publisher:
            asyncio.create_task(self._publish_rows_async(rows, instId, tf))
            
        if self.feature_writer is not None:
            updates_msg = f"InstId: {instId}, tf: {tf}, total {self.engine.updates_cnt} updates in this slot."
            logger.info(updates_msg)
            self.feature_writer.add(feats)
            
        return feats if not feats.empty else None

    def _extract_tf(self, channel: str):
        m = self._tf_pat.search(channel)
        return m.group(1) if m else None
    
    def _collect_bar_close_rows(self, feats: "pd.DataFrame", instId: str, tf: str) -> list[dict]:
        """
        仅做两件事：
        1) 按 ts 升序，保证乱序到达时的正确发出顺序；
        2) 以 (instId, tf) 维度做“单调递增去重”（ts <= last_ts 的丢弃）。
        返回：应当产出的 DataFrame 行（dict 列表），供后续 build_snapshot_from_row 使用。
        """
        if feats is None or feats.empty:
            return []
        key = (instId, tf)
        last_ts = self._emit_last_ts.get(key)

        out_rows: list[dict] = []
        for _, r in feats.sort_values("ts").iterrows():
            ts = int(r["ts"])
            if last_ts is not None and ts <= last_ts:
                continue
            out_rows.append(r.to_dict())
            last_ts = ts  # 更新到最新已选择的 ts

        self._emit_last_ts[key] = last_ts
        return out_rows
    
    async def _publish_rows_async(self, rows: list[dict], instId: str, tf: str):
        ts_now = int(asyncio.get_running_loop().time() * 1000)  # 近似
        for snap in rows:
            payload = {
                "kind": "FeaturesUpdated",
                "inst": instId,
                "tf": tf,
                "ts_close": snap.get("ts_close") or snap.get("ts"),
                "features": snap.get("features") or snap,
                "feature_version": getattr(self.engine, "feature_version", "v1.0.0"),
                "engine_id": getattr(self.engine, "engine_id", "fe-worker"),
                "computed_at": ts_now,
                "trace_id": snap.get("trace_id"),
                "seq": snap.get("seq"),
            }
            await self.publisher.publish(self.stream_name, payload)