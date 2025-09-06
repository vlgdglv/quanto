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
# {
#   "snapshot": {
#     "last_price": 112238,
#     "atr": 105.4,
#     "rv_ewma": 0.0,
#     "spread_bp": 0.01,
#     "funding_rate": 0.0001,
#     "funding_premium_z": -0.000217,
#     "funding_time_to_next_min": 858.0,
#     "oi": 2734128.43,
#     "d_oi_rate": 1.3e-05
#   },
#   "trend_momentum": {
#     "ema_fast": 112238,
#     "ema_slow": 112238,
#     "macd_dif": 0.0,
#     "macd_hist": 0.0,
#     "rsi": 50.0,
#     "s_mom_slope_H60m": 0.0,
#     "s_mom_slope_H180m": 0.0,
#     "s_mom_slope_H420m": 0.0,
#     "s_rsi_mean_H60m": 50.0,
#     "s_rsi_std_H60m": 0.0089,
#     "...": "..."
#   },
#   "microstructure": {
#     "ofi_5s": -70.37,
#     "s_ofi_sum_30m": -70.37,
#     "cvd": 801.79,
#     "s_cvd_delta_H60m": 0.0,
#     "s_spread_bp_mean_H60m": 0.0089
#   },
#   "volatility_regime": {
#     "s_squeeze_on_dur": 1.0,
#     "donchian_width_norm": 0.367,
#     "s_donchian_dist_upper": 1023.2,
#     "s_donchian_dist_lower": 511.4
#   },
#   "positioning": {
#     "s_oi_rate_H60m": 0.0,
#     "s_oi_rate_H180m": 0.0,
#     "s_oi_rate_H420m": 0.0
#   }
# }


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