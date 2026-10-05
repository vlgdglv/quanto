# feature/integrate.py
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import pandas as pd

import math
from utils.logger import logger
from copy import deepcopy

HORIZON_MIN = [60, 180, 420]
UNIVERSE = ["ETH-USDT-SWAP", "DOGE-USDT-SWAP"]

def _to_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default

def _to_int(x, default: int = 0) -> int:
    try:
        if x is None:
            return default
        v = int(x)
        return v
    except Exception:
        return default
    
CORE_GROUPS = ["snapshot", "trend_momentum", "microstructure", "volatility_regime", "positioning"]

def build_snapshot_from_row(
    row: dict,
    *,
    state: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    def _to_int(x): 
        try: return int(x) if x is not None else None
        except: return None
    def _to_float(x, default=None):
        try: return float(x) if x is not None else default
        except: return default
    def _get(d, k, default=None):
        v = d.get(k)
        return v if v is not None else default

    instId = row.get("instId")
    tf = row.get("tf")

    snap: Dict[str, Any] = {
        "instId": instId,
        "tf": tf,
        "ts": _to_int(row.get("ts")),
        "universe": row.get("universe") or ["ETH-USDT-SWAP", "DOGE-USDT-SWAP"],
        "horizon_min": row.get("horizon_min") or [60, 180, 420],

        "snapshot": {
            "last_price": _to_float(row.get("c")),
            "atr": _to_float(row.get("atr")),
            "rv_ewma": _to_float(row.get("rv_ewma")),
            "spread_bp": _to_float(row.get("spread_bp")),
            "funding_rate": _to_float(row.get("funding_rate")),
            "funding_premium_z": _to_float(row.get("funding_premium_z")),
            "funding_time_to_next_min": _to_float(row.get("funding_time_to_next_min")),
            "oi": _to_float(row.get("oi")),
            "d_oi_rate": _to_float(row.get("d_oi_rate")),
        },

        "trend_momentum": {
            "ema_fast": _to_float(row.get("ema_fast")),
            "ema_slow": _to_float(row.get("ema_slow")),
            "macd_dif": _to_float(row.get("macd_dif")),
            "macd_hist": _to_float(row.get("macd_hist")),
            "rsi": _to_float(row.get("rsi"), 50.0),
            "s_mom_slope_H60m": _to_float(row.get("s_mom_slope_H60m")),
            "s_mom_slope_H180m": _to_float(row.get("s_mom_slope_H180m")),
            "s_mom_slope_H420m": _to_float(row.get("s_mom_slope_H420m")),
            "s_rsi_mean_H60m": _to_float(row.get("s_rsi_mean_H60m"), 50.0),
            "s_rsi_std_H60m": _to_float(row.get("s_rsi_std_H60m")),
            "s_rsi_mean_H180m": _to_float(row.get("s_rsi_mean_H180m"), 50.0),
            "s_rsi_std_H180m": _to_float(row.get("s_rsi_std_H180m")),
            "s_rsi_mean_H420m": _to_float(row.get("s_rsi_mean_H420m"), 50.0),
            "s_rsi_std_H420m": _to_float(row.get("s_rsi_std_H420m")),
        },

        "microstructure": {
            "ofi_5s": _to_float(row.get("ofi_5s")),
            "s_ofi_sum_30m": _to_float(row.get("s_ofi_sum_30m")),
            "cvd": _to_float(row.get("cvd")),
            "s_cvd_delta_H60m": _to_float(row.get("s_cvd_delta_H60m")),
            "s_spread_bp_mean_H60m": _to_float(row.get("s_spread_bp_mean_H60m")),
        },

        "volatility_regime": {
            "s_squeeze_on_dur": _to_float(row.get("s_squeeze_on_dur")),
            "donchian_width_norm": _to_float(row.get("donchian_width_norm")),
            "s_donchian_dist_upper": _to_float(row.get("s_donchian_dist_upper")),
            "s_donchian_dist_lower": _to_float(row.get("s_donchian_dist_lower")),
        },

        "positioning": {
            "s_oi_rate_H60m": _to_float(row.get("s_oi_rate_H60m")),
            "s_oi_rate_H180m": _to_float(row.get("s_oi_rate_H180m")),
            "s_oi_rate_H420m": _to_float(row.get("s_oi_rate_H420m")),
        },
    }

    extras_micro = {}
    if "kyle_lambda" in row: extras_micro["kyle_lambda"] = _to_float(row.get("kyle_lambda"))
    if "vpin" in row: extras_micro["vpin"] = _to_float(row.get("vpin"))
    if extras_micro:
        snap["microstructure"]["extra"] = extras_micro

    extras_trend = {}
    if "s_macd_pos_streak" in row: extras_trend["s_macd_pos_streak"] = _to_float(row.get("s_macd_pos_streak"))
    if "s_macd_neg_streak" in row: extras_trend["s_macd_neg_streak"] = _to_float(row.get("s_macd_neg_streak"))
    if extras_trend:
        snap["trend_momentum"]["extra"] = extras_trend

    # ===== 新增：可选 state/constraints/meta 透传（不做数值加工）=====
    if state is not None:
        snap["state"] = deepcopy(state)
    if constraints is not None:
        snap["constraints"] = deepcopy(constraints)
    if meta is not None:
        snap["meta"] = deepcopy(meta)

    # ===== 新增：data_quality / availability（不改变任何数值，仅告知可用性）=====
    data_quality = {}
    availability = {}
    for grp in CORE_GROUPS:
        g = snap.get(grp, {}) or {}
        present = [k for k, v in g.items() if v is not None]
        missing = [k for k, v in g.items() if v is None]
        availability[grp] = present
        data_quality[grp] = {
            "missing_count": len(missing),
            "missing_fields": missing
        }
    snap["data_quality"] = data_quality
    snap["availability"] = availability

    return snap


@dataclass
class _KeyState:
    last_emit_ts: Optional[int] = None
    last_seen_ts: Optional[int] = None
