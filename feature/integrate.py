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

    # ===== 新增：llm_config（仅协议，不计算）=====
    snap["llm_config"] = {
        "scoring_weights": {"trend": 0.35, "flow": 0.35, "vol": 0.15, "pos": 0.15},
        "decision_thresholds": {
            "strong_long": 0.60,
            "weak_long": 0.20,
            "hold_hi": 0.20,
            "weak_short": -0.20,
            "strong_short": -0.60
        },
        "risk_guardrails": {
            "min_liq_buffer_pct": 20.0,
            "spread_bp_pctile_block": 0.90,
            "rr_min_when_vol_expansion": 2.5
        }
    }

    return snap

def dep_build_snapshot_from_row(row: dict) -> Dict[str, Any]:
    """
    将一行 features（来自 FeatureEnginePD.update_candles 的输出）转换为 LLM 输入快照（新 schema）。
    约定：row['ts'] 为该 bar 的“收盘时刻”毫秒。
    面向 1h–7h 短线（ETH/DOGE 永续），只暴露与决策高度相关的统计量。
    """
    instId = row.get("instId")
    tf = row.get("tf")

    snap: Dict[str, Any] = {
        # 元信息
        "instId": instId,
        "tf": tf,
        "ts": _to_int(row.get("ts")),
        "universe": UNIVERSE,
        "horizon_min": HORIZON_MIN,

        # 市场横截面 snapshot（当前时点/近期直接量）
        "snapshot": {
            "last_price": _to_float(row.get("c")),  # 收盘价作为 last_price
            "atr": _to_float(row.get("atr")),
            "rv_ewma": _to_float(row.get("rv_ewma")),
            "spread_bp": _to_float(row.get("spread_bp")),
            "funding_rate": _to_float(row.get("funding_rate")),
            "funding_premium_z": _to_float(row.get("funding_premium_z")),
            "funding_time_to_next_min": _to_float(row.get("funding_time_to_next_min")),
            "oi": _to_float(row.get("oi")),
            "d_oi_rate": _to_float(row.get("d_oi_rate")),
        },

        # 趋势与动量（兼顾 1–7 小时窗）
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

        # 微观结构（短线最敏感）
        "microstructure": {
            "ofi_5s": _to_float(row.get("ofi_5s")),
            "s_ofi_sum_30m": _to_float(row.get("s_ofi_sum_30m")),
            "cvd": _to_float(row.get("cvd")),
            "s_cvd_delta_H60m": _to_float(row.get("s_cvd_delta_H60m")),
            "s_spread_bp_mean_H60m": _to_float(row.get("s_spread_bp_mean_H60m")),
        },

        # 波动/区间状态（识别 squeeze 与通道边界）
        "volatility_regime": {
            "s_squeeze_on_dur": _to_float(row.get("s_squeeze_on_dur")),
            "donchian_width_norm": _to_float(row.get("donchian_width_norm")),
            "s_donchian_dist_upper": _to_float(row.get("s_donchian_dist_upper")),
            "s_donchian_dist_lower": _to_float(row.get("s_donchian_dist_lower")),
        },

        # 持仓/定位（资金与仓位的节奏）
        "positioning": {
            "s_oi_rate_H60m": _to_float(row.get("s_oi_rate_H60m")),
            "s_oi_rate_H180m": _to_float(row.get("s_oi_rate_H180m")),
            "s_oi_rate_H420m": _to_float(row.get("s_oi_rate_H420m")),
        },
    }

    # —— 可选增强：若存在以下列，则自动并入（对 ETH/DOGE 某些交易所数据有用）——
    # 1) kyle_lambda / vpin 代表冲击成本与流动性不对称，可放入 microstructure.extra
    extras_micro = {}
    if "kyle_lambda" in row:
        extras_micro["kyle_lambda"] = _to_float(row.get("kyle_lambda"))
    if "vpin" in row:
        extras_micro["vpin"] = _to_float(row.get("vpin"))
    if extras_micro:
        snap["microstructure"]["extra"] = extras_micro

    # 2) 若你需要 MACD 正负连击作为趋势延续特征（已在列中提供），则放入 trend_momentum.extra
    extras_trend = {}
    if "s_macd_pos_streak" in row:
        extras_trend["s_macd_pos_streak"] = _to_float(row.get("s_macd_pos_streak"))
    if "s_macd_neg_streak" in row:
        extras_trend["s_macd_neg_streak"] = _to_float(row.get("s_macd_neg_streak"))
    if extras_trend:
        snap["trend_momentum"]["extra"] = extras_trend

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