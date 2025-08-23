# feature/integrate.py
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import pandas as pd
from datafeed.handlers import channel_registry  # 你给的 register_channel 映射
from feature.engine_pd import FeatureEnginePD

from utils.logger import logger

SNAPSHOT_FIELDS = [
    "ema_fast","ema_slow","macd_dif","macd_dea","macd_hist",
    "rsi","kdj_k","kdj_d","kdj_j",
    "atr","rv_ewma","spread_bp","ofi_5s","obi_5"
]


def build_snapshot_from_row(row: dict) -> Dict[str, Any]:
    """
    将一行 features（来自 FeatureEnginePD.update_candles 的输出）转换为 LLM 输入快照。
    约定：row['ts'] 为该 bar 的“收盘时刻”毫秒。
    """
    snap = {
        "instId": row["instId"],
        "tf": row["tf"],
        "ts": int(row["ts"]),
        "trend": {
            "ema_fast": float(row.get("ema_fast", 0.0)),
            "ema_slow": float(row.get("ema_slow", 0.0)),
            "macd_dif": float(row.get("macd_dif", 0.0)),
            "macd_dea": float(row.get("macd_dea", 0.0)),
            "macd_hist": float(row.get("macd_hist", 0.0)),
            "rsi": float(row.get("rsi", 50.0)),
            "kdj_k": float(row.get("kdj_k", 50.0)),
            "kdj_d": float(row.get("kdj_d", 50.0)),
            "kdj_j": float(row.get("kdj_j", 50.0)),
        },
        "volatility": {
            "atr": float(row.get("atr", 0.0)),
            "rv_ewma": float(row.get("rv_ewma", 0.0)),
        },
        "micro": {
            "spread_bp": float(row.get("spread_bp", 0.0)),
            "ofi_5s": float(row.get("ofi_5s", 0.0)),
        },
    }
    if "obi_5" in row:
        snap["micro"]["obi_5"] = float(row.get("obi_5") or 0.0)
    return snap

@dataclass
class _KeyState:
    last_emit_ts: Optional[int] = None
    last_seen_ts: Optional[int] = None


class SnapshotThrottler:
    """
    控制快照产出频率与去重：
    - 对每个 (instId, tf) 至多每 min_interval_ms 产出一份快照；
    - 同一 ts 不重复产出。
    - 可选对齐 5 分钟自然边界（align_to_5m）。
    """
    def __init__(self, min_interval_ms: int = 300_000, 
                 align_to_5m: bool = False):
        self.min_interval_ms = int(min_interval_ms)
        self.align_to_5m = align_to_5m
        self._state: Dict[Tuple[str, str], _KeyState] = {}
        logger.info(f"SnapshotThrottler init min_interval_ms={min_interval_ms} align_to_5m={align_to_5m}")

    def _should_align(self, ts: int) -> bool:
        if not self.align_to_5m:
            return True
        return ts % 300_000 == 0
    
    def ingest_features_df(self, features_df: pd.DataFrame, policy: str = "bar_close") -> List[Dict]:
        """
        输入：features DataFrame（可能包含多行、多 instId）
        输出：应当产出的 snapshot 列表（每个为 dict）
        policy:
          - "bar_close": 收盘行必发（不受 min_interval 限制），仅去重与可选对齐
          - "interval":  按最小间隔限流（旧逻辑）
        """
        if features_df is None or features_df.empty:
            return []
        
        out: List[dict] = []
        for _, r in features_df.sort_values("ts").iterrows():
            instId, tf, ts = r["instId"], r["tf"], int(r["ts"])
            key = (instId, tf)
            state = self._state.setdefault(key, _KeyState())

            if state.last_seen_ts is not None and ts <= state.last_seen_ts:
                continue
            state.last_seen_ts = ts
            
            # if policy == "interval":
            #     if state.last_emit_ts is not None and (ts - state.last_emit_ts) < self.min_interval_ms:
            #         continue
            #     if not self._should_align(ts):
            #         continue
            # else:  # "bar_close"
            #     if not self._should_align(ts):
            #         continue  # 仅按对齐/去重限制
            
            out.append(r.to_dict())
            state.last_emit_ts = ts
        
        return out