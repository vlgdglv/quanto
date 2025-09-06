# feature/engine_pd.py
from __future__ import annotations
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Deque, Tuple, Sequence, List, Any, Callable
import numpy as np
import pandas as pd
from datetime import datetime
import math, time
import re
from fnmatch import fnmatch

from feature.summarizer import FeatureSummarizer
from utils.logger import logger

def ts_to_str(ts):
    dt = datetime.fromtimestamp(ts/1000)
    return dt.strftime("%Y%m%d%H%M%S")

def _to_ts_ms(x: Any) -> Optional[int]:
    """
    将输入转换为 UTC 毫秒时间戳。
    支持：int(ms), float(ms), pandas.Timestamp, ISO8601 字符串。
    """
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if isinstance(x, (int, np.integer)):
        # 假定是 ms
        return int(x)
    if isinstance(x, float):
        # 假定是 ms
        return int(x)
    if isinstance(x, pd.Timestamp):
        if x.tz is None:
            x = x.tz_localize("UTC")
        return int(x.value // 1_000_000)  # ns -> ms
    if isinstance(x, str):
        # 尝试解析
        try:
            ts = pd.to_datetime(x, utc=True, errors="coerce")
            if pd.isna(ts):
                return None
            return int(ts.value // 1_000_000)
        except Exception:
            return None
    return None

def _safe_float(x: Any, default=np.nan) -> float:
    try:
        if x is None: return default
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        return float(str(x))
    except Exception:
        return default

def _clip01(x: float) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    return max(0.0, min(1.0, x))

def _minutes(ms: int) -> float:
    return ms / 60000.0

def _floor_to_freq(ts_ms: int, freq: str) -> int:
    """
    将 ts_ms 向下取整到频率边界（支持 '15m','30m','1h'）。
    """
    if ts_ms is None: return None
    epoch = ts_ms // 1000
    if freq.endswith('m'):
        m = int(freq[:-1])
        # 以分钟为粒度对齐
        floored = (epoch // (60*m)) * (60*m)
        return floored * 1000
    if freq.endswith('h'):
        h = int(freq[:-1])
        floored = (epoch // (3600*h)) * (3600*h)
        return floored * 1000
    # 默认不变
    return ts_ms

round_map = {
    # —— 趋势与均线（价格维度）——
    "ema_fast": 6,
    "ema_slow": 6,
    "macd_dif": 6,
    "macd_dea": 6,
    "macd_hist": 6,

    # —— 震荡指标 —— 
    "rsi": 2,           # 0~100
    "kdj_k": 2,
    "kdj_d": 2,
    "kdj_j": 2,

    # —— 波动/方差（价格/收益维度）——
    "atr": 6,           # 价格单位
    "rv_ewma": 8,       # 建议更细一些，收益方差通常很小

    # —— 微结构（tick 级）——
    "spread_bp": 2,     # 基点
    "ofi_5s": 3,        # 体量可能为小数，3 位兼顾
    "qi1": 3,           # [-1,1]
    "qi5": 3,           # [-1,1]
    "microprice": 6,    # 价格

    "cvd": 3,           # 累计主动量
    "kyle_lambda": 6,   # 冲击系数一般很小
    "vpin": 3,          # [0,1] 附近

    # —— Squeeze & Donchian —— 
    "squeeze_ratio": 3,         # BB/KC 宽度比
    # squeeze_on 为 bool，不做 round
    "donchian_upper": 6,        # 价格
    "donchian_lower": 6,        # 价格
    "donchian_width": 6,        # 价格
    "donchian_width_norm": 3,   # 归一化宽度

    # —— 效率比 —— 
    "er": 3,             # [0,1] 内

    # —— Funding —— 
    "funding_rate": 6,         # 通常很小（~1e-4）
    "funding_rate_ema": 6,
    "funding_premium": 6,      # premium/premium_index
    "funding_premium_ema": 6,
    "funding_premium_z": 3,    # z-score
    "funding_annualized": 4,   # 年化比率，略细一点
    # funding_time / next_funding_time 为字符串或时间，不 round
    "funding_time_to_next_min": 2,  # 分钟

    # —— OI —— 
    "oi": 2,             # 不同交易所返回可能带小数，2 位实用
    "oiCcy": 3,          # 币计 OI
    "oiUsd": 2,          # 美元计 OI
    "d_oi": 2,           # 变化量
    "d_oi_rate": 6,      # 相对变化率，通常很小
    "oi_ema": 2,         # OI 的 EMA

     # —— 新增的非 H 维度列 ——
    "s_macd_pos_streak": 0,          # 连续为正的 DIF-DEA 段数（或 bar 数），取整
    "s_macd_neg_streak": 0,          # 同上
    "s_squeeze_on_dur": 0,           # squeeze 持续时长，若单位为 bar 也应取整
    "s_donchian_dist_upper": 6,      # 与上轨的价格距离
    "s_donchian_dist_lower": 6,      # 与下轨的价格距离
    "s_donchian_mid_dev": 6,         # 相对中线的偏差，价格单位
    "s_ofi_sum_30m": 3,              # OFI 汇总，通常取 3 位足够

    # —— 带 H{H}m 后缀的通配列（glob） ——
    "s_mom_slope_H*m": 6,            # 动量斜率：通常是价格差/单位时间
    "s_rsi_mean_H*m": 2,             # RSI 均值（0~100）
    "s_rsi_std_H*m": 2,              # RSI 标准差，典型量级较小也可 2
    "s_spread_bp_mean_H*m": 2,       # 基点均值
    "s_cvd_delta_H*m": 3,            # CVD 的变化量
    "s_kyle_ema_H*m": 6,             # Kyle λ 量级很小
    "s_vpin_mean_H*m": 3,            # [0,1] 周期均值
    "s_oi_rate_H*m": 6,              # OI 变化率，通常很小
}


def round_numeric_columns(df, round_specs, do_round=True):
    """
    round_specs 可为:
      - dict[str, int]: 兼容你现在的写法（键为精确名或包含通配符的 glob）
      - 或 List[Tuple[str, int]]: 想要固定优先级时可用 list 保序
        建议顺序：精确 > 通配 > 正则
      - 约定：若 key 形如 r"^regex$" 或以 "re:" 开头，则视作正则
    """
    if not do_round or df is None or df.empty:
        return df

    # 统一为有序列表（保持用户传入顺序）
    if isinstance(round_specs, dict):
        items = list(round_specs.items())
    else:
        items = list(round_specs)

    # 记录已经被处理过的列，防止被重复覆盖
    processed = set()

    def is_regex_key(k: str) -> bool:
        return k.startswith("re:") or (k.startswith("^") and k.endswith("$"))

    # 三轮：精确 -> glob -> regex
    # 1) 精确匹配
    for key, nd in items:
        if is_regex_key(key) or any(ch in key for ch in "*?"):
            continue
        if key in df.columns and key not in processed:
            df[key] = pd.to_numeric(df[key], errors="coerce").round(nd)
            processed.add(key)

    # 2) glob 匹配
    for key, nd in items:
        if is_regex_key(key) or not any(ch in key for ch in "*?"):
            continue
        for col in df.columns:
            if col in processed:
                continue
            if fnmatch(col, key):
                df[col] = pd.to_numeric(df[col], errors="coerce").round(nd)
                processed.add(col)

    # 3) 正则匹配
    for key, nd in items:
        if not is_regex_key(key):
            continue
        pat = key[3:] if key.startswith("re:") else key
        rx = re.compile(pat)
        for col in df.columns:
            if col in processed:
                continue
            if rx.fullmatch(col):
                df[col] = pd.to_numeric(df[col], errors="coerce").round(nd)
                processed.add(col)

    return df


# -----------
# 
# -----------

@dataclass
class EMAState:
    n: int
    alpha: float
    value: Optional[float] = None

@dataclass
class MACDState:
    ema_fast: EMAState
    ema_slow: EMAState
    dea:      EMAState
    diff:     Optional[float] = None
    hist:     Optional[float] = None

def _alpha(n: int) -> float:
    return 2.0 / (n + 1.0)

def macd_init(fast=12, slow=26, dea_n=9) -> MACDState:
    return MACDState(
        ema_fast=EMAState(n=fast, alpha=_alpha(fast)),
        ema_slow=EMAState(n=slow, alpha=_alpha(slow)),
        dea=EMAState(n=dea_n, alpha=_alpha(dea_n))
    )

@dataclass
class RSIState:
    n: int = 14
    avg_gain: Optional[float] = None
    avg_loss: Optional[float] = None
    last_close: Optional[float] = None
    rsi: Optional[float] = None

@dataclass
class ATRState:
    n: int = 14
    atr: Optional[float] = None
    prev_close: Optional[float] = None

@dataclass
class EWVarState:
    lam: float = 0.94
    mean: Optional[float] = None
    var: Optional[float] = None

class _MonoMax:
    def __init__(self):
        self.q: Deque[Tuple[int, float]] = deque()

    def push(self, i, v):
        while self.q and self.q[-1][1] <= v:
            self.q.pop()
        self.q.append((i, v))

    def pop_until(self, lo):
        while self.q and self.q[0][0] < lo:
            self.q.popleft()
    
    def max(self):
        return self.q[0][1] if self.q else np.nan
    
class _MonoMin:
    def __init__(self):
        self.q: Deque[Tuple[int, float]] = deque()

    def push(self, i, v):
        while self.q and self.q[-1][1] >= v:
            self.q.pop()
        self.q.append((i, v))

    def pop_until(self, lo):
        while self.q and self.q[0][0] < lo:
            self.q.popleft()
    
    def min(self):
        return self.q[0][1] if self.q else np.nan
    
@dataclass
class KDJState:
    n: int = 9
    k: float = 50.0
    d: float = 50.0
    j: float = 50.0
    i: int = 0
    qmax: _MonoMax = _MonoMax()
    qmin: _MonoMin = _MonoMin()

@dataclass
class MicroState:
    spread_bp: float = np.nan # books(best_bid/best_ask) spread in bps
    ofi_5s: float = 0.0 # trades window of 5s order flow imbalance

# ----------------
#
# ----------------
def ema_update(state: EMAState, x: float) -> float:
    if state.value is None:
        state.value = x
    else:
        state.value = state.alpha * x + (1.0 - state.alpha) * state.value
    return state.value

def macd_update(state: MACDState, close: float):
    ef = ema_update(state.ema_fast, close)
    es = ema_update(state.ema_slow, close)
    state.diff = ef - es
    dea = ema_update(state.dea, state.diff)
    state.hist = 2.0 * (state.diff - dea)
    return state.diff, state.dea, state.hist

def rsi_update(state: RSIState, close: float):
    if state.last_close is None:
        state.last_close = close
        state.rsi = 50.0
        return state.rsi
    delta = close - state.last_close
    gain, loss = max(delta, 0.0), max(-delta, 0.0)
    if state.avg_gain is None or state.avg_loss is None:
        state.avg_gain, state.avg_loss = gain, loss
    else:
        n = state.n
        state.avg_gain = (state.avg_gain * (n - 1.0) + gain) / n
        state.avg_loss = (state.avg_loss * (n - 1.0) + loss) / n
    state.last_close = close
    if state.avg_loss == 0.0:
        state.rsi = 100.0
    else:
        rs = state.avg_gain / state.avg_loss
        state.rsi = 100.0 - (100.0 / (1.0 + rs))
    return state.rsi

def atr_update(state: ATRState, high: float, low: float, close: float):
    if state.prev_close is None:
        state.atr = high - low
    else:
        tr = max(high - low, abs(high - state.prev_close), abs(low - state.prev_close))
        if state.atr is None:
            state.atr = tr
        else:
            n = state.n
            state.atr = (state.atr * (n - 1.0) + tr) / n
    state.prev_close = close
    return state.atr

def kdj_update(state: KDJState, high: float, low: float, close: float,
               k_smooth=3, d_smooth=3):
    state.qmax.push(state.i, high)
    state.qmax.pop_until(state.i - state.n + 1)
    state.qmin.push(state.i, low)
    state.qmin.pop_until(state.i - state.n + 1)
    state.i += 1
    Hn, Ln = state.qmax.max(), state.qmin.min()
    rsv = 50.0 if Hn == Ln else (close - Ln) / (Hn - Ln) * 100.0
    ka, da = 1.0 / k_smooth, 1.0 / d_smooth
    state.k = (1 - ka) * state.k + ka * rsv
    state.d = (1 - da) * state.d + da * state.k
    state.j = 3.0 * state.k - 2.0 * state.d
    return state.k, state.d, state.j

def ewma_var_update(state: EWVarState, x: float) -> float:
    if state.mean is None:
        state.mean, state.var = x, 0.0
    else:
        m_prev = state.mean
        state.mean = state.lam * state.mean + (1.0 - state.lam) * x
        state.var = state.lam * state.var + (1.0 - state.lam) * (x - m_prev) * (x - state.mean)
    return max(state.var or 0.0, 0.0)

@dataclass
class QIMicropriceState:
    """基于 L1/L5 的队列不平衡与 microprice"""
    qi1: float = np.nan
    qi5: float = np.nan
    microprice: float = np.nan  # 仅用 L1 价格与数量计算
    last_b1:  Optional[Tuple[float,float]] = None
    last_a1:  Optional[Tuple[float,float]] = None

def _sum_depth(levels: Sequence[Tuple[float, float]], n: int) -> Tuple[float, float]:
    qty_sum, wsum = 0.0, 0.0
    for i in range(min(n, len(levels))):
        p, q = float(levels[i][0]), float(levels[i][1])
        if q <= 0 or p <= 0:
            continue
        qty_sum += q
        wsum += p * q
    return qty_sum, wsum

def qi_microprice_update(state: QIMicropriceState, 
                         bids: Sequence[Tuple[float, float]],
                         asks: Sequence[Tuple[float, float]]):
    """
    bids/asks: 形如 [(price, size), ...]，传入前5档即可（若不足按实际长度）
    计算：
      qi1 = (Vb1 - Va1) / (Vb1 + Va1)
      qi5 = (ΣVb1..5 - ΣVa1..5) / (ΣVb1..5 + ΣVa1..5)
      microprice = (a1*Vb1 + b1*Va1) / (Vb1 + Va1)
    """
    if not bids or not asks:
        state.qi1 = state.qi5 = state.microprice = np.nan
        return state.qi1, state.qi5, state.microprice
    
    b1_p, b1_q = float(bids[0][0]), float(bids[0][1])
    a1_p, a1_q = float(asks[0][0]), float(asks[0][1])

    denom1 = b1_q + a1_q
    state.qi1 = (b1_q - a1_q) / denom1 if denom1 > 0 else np.nan

    vb5, _ = _sum_depth(bids, 5)
    va5, _ = _sum_depth(asks, 5)
    denom5 = vb5 + va5
    state.qi5 = (vb5 - va5) / denom5 if denom5 > 0 else np.nan

    denom_m = b1_q + a1_q
    state.microprice = (a1_p * b1_q + b1_p * a1_q) / denom_m if denom_m > 0 else (b1_p + a1_p) / 2.0
    state.last_b1 = (b1_p, b1_q)
    state.last_a1 = (a1_p, a1_q)

    return state.qi1, state.qi5, state.microprice

# ========= CVD =========
@dataclass
class CVDState:
    """累计主动买卖差；可作为多尺度 CVD 的基础内核"""
    cvd: float = 0.0

def cvd_update(state: CVDState, signed_size: float) -> float:
    """
    signed_size: 主动买为 +sz，主动卖为 -sz
    """
    state.cvd += float(signed_size)
    return state.cvd


# ========= Kyle's λ（EW 回归） =========
@dataclass
class KyleLambdaState:
    """
    价格冲击系数（Δmid ~ λ * q），EW（指数加权）在线回归
    """
    alpha: float = 0.1
    sxx: float = 1e-8
    sxy: float = 0.0
    value: float = 0.0
    last_mid: Optional[float] = None

def kyle_lambda_update(state: KyleLambdaState, 
                       mid: float, 
                       signed_size:float) -> float:
    """
    输入：
      mid: 当前中间价（(best_bid+best_ask)/2）
      signed_size: 当前时点或极短窗的净主动量（买为+，卖为-）
    输出：
      state.value: 估计的 λ
    """
    if state.last_mid is None:
        state.last_mid = float(mid)
        return state.value
    
    dm = float(mid) - state.last_mid
    q = float(signed_size)

    a = state.alpha
    state.sxx = (1 - a) * state.sxx + a * (q * q)
    state.sxy = (1 - a) * state.sxy + a * (q * dm)
    state.value = state.sxy / max(state.sxx, 1e-12)

    state.last_mid = float(mid)
    return state.value

# ========= VPIN =========
@dataclass
class VPINState:
    """
    体积同步信息概率近似：
      - bucket_vol: 每桶目标体积（与合约面值一致的量纲，建议用名义价值或张数）
      - window: 近 N 桶的滚动均值
    """
    bucket_vol: float = 10_000.0
    window: int = 50
    cur_buy: float = 0.0
    cur_sell: float = 0.0
    filled_in_bucket: float = 0.0
    last_vals: Optional[Deque[float]] = None
    vpin: float = np.nan

def vpin_update(state: VPINState, signed_size: float)->float:
    """
    输入 signed_size：主动买 +sz / 主动卖 -sz（与 CVD 同口径）
    逻辑：将绝对体积灌入体积桶；每满一桶，产出一个 |B−S|/V 的值，并更新滚动均值。
    """
    if state.last_vals is None:
        state.last_vals = deque(maxlen=state.window)

    vol = abs(float(signed_size))
    if vol == 0.0:
        return state.vpin
    
    side_buy = signed_size > 0

    while vol > 0.0:
        remain = max(state.bucket_vol - state.filled_in_bucket, 1e-12)
        take = min(vol, remain)

        if side_buy:
            state.cur_buy += take
        else:
            state.cur_sell += take
        state.filled_in_bucket += take
        vol -= take

        if state.filled_in_bucket >= state.bucket_vol - 1e-12:
            imb = abs(state.cur_buy - state.cur_sell) / max(state.bucket_vol, 1e-12)
            state.last_vals.append(imb)

            overflow = state.filled_in_bucket - state.bucket_vol
            if overflow > 0:
                if side_buy:
                    state.cur_buy = overflow
                    state.cur_sell = 0.0
                else:
                    state.cur_sell = overflow
                    state.cur_buy = 0.0
            else:
                state.cur_buy = 0.0
                state.cur_sell = 0.0
            state.filled_in_bucket = overflow
    if state.last_vals:
        state.vpin = float(np.mean(state.last_vals))
    return state.vpin

# ========= Squeeze（BB / Keltner） =========
@dataclass
class SqueezeState:
    """
    使用 EW 方差估计 BB 宽度；Keltner 使用 ATR
      - bb_k: BB 标准差倍数（常用 2）
      - kc_k: Keltner ATR 倍数（常用 1.5）
      - lam:  EW 方差的 λ（越小越灵敏，与你 EWVarState 吻合）
    """
    bb_k: float = 2.0
    kc_k: float = 1.5
    lam: float = 0.94
    ewvar: "EWVarState" = field(default_factory=lambda: EWVarState(lam=0.94))
    ratio: float = np.nan
    squeeze_on: Optional[bool] = None

def squeeze_update(state: SqueezeState, 
                   atr_state: ATRState,
                   close: float):
    """
    返回 (ratio, squeeze_on)
      ratio = BBWidth / KCWidth
      squeeze_on: ratio < 1（BB 被 Keltner“夹住”）
    """
    state.ewvar.lam = state.lam
    var = ewma_var_update(state.ewvar, float(close))
    std = np.sqrt(max(var, 0.0))
    bb_width = 2.0 * state.bb_k * std

    atr = (atr_state.atr or 0.0) if atr_state is not None else 0.0
    kc_width = 2.0 * state.kc_k * atr

    state.ratio = (bb_width / kc_width) if kc_width > 0  else np.nan
    state.squeeze_on = (state.ratio < 1.0) if np.isfinite(state.ratio) else None
    return state.ratio, state.squeeze_on


# ========= Donchian =========
@dataclass
class DonchianState:
    """
    Donchian 通道：近 n 根的最高/最低；输出宽度与 ATR 规范化宽度
    """
    n: int = 20
    i: int = 0
    qmax: _MonoMax = _MonoMax()
    qmin: _MonoMin = _MonoMin()
    upper: float = np.nan
    lower: float = np.nan
    width: float = np.nan
    width_norm: float = np.nan

def donchian_update(state: DonchianState, 
                    high: float, low:float,
                    atr_state: ATRState):
    state.qmax.push(state.i, float(high))
    state.qmax.pop_until(state.i - state.n + 1)
    state.qmin.push(state.i, float(low))
    state.qmin.pop_until(state.i - state.n + 1)
    state.i += 1

    Hn, Ln = state.qmax.max(), state.qmin.min()
    state.upper, state.lower = Hn, Ln
    state.width = (Hn - Ln) if (np.isfinite(Hn) and np.isfinite(Ln)) else np.nan
    atr = atr_state.atr or 0.0
    state.width_norm = (state.width / atr) if atr > 0 else np.nan
    return state.upper, state.lower, state.width, state.width_norm

# ========= ER（Efficiency Ratio） =========
@dataclass
class ERState:
    """
    ER = |close_t - close_{t-n}| / Σ_{k=1..n} |close_k - close_{k-1}|
    使用长度为 n 的步长绝对变化和
    """
    n: int = 10
    closes: Optional[Deque[float]] = None
    deltas: Optional[Deque[float]] = None
    sum_abs: float = 0.0
    last_close: Optional[float] = None
    er: float = np.nan

def er_update(state: ERState, close: float) -> float:
    if state.closes is None:
        state.closes = deque()
    if state.deltas is None:
        state.deltas = deque()

    c = float(close)
    if state.last_close is not None:
        d = abs(c - state.last_close)
        state.deltas.append(d)
        state.sum_abs += d
        if len(state.deltas) > state.n:
            old = state.deltas.popleft()
            state.sum_abs -= old
    
    state.closes.append(c)
    if len(state.closes) > state.n + 1:
        state.closes.popleft()

    if len(state.closes) >= state.n + 1 and state.sum_abs > 0:
        net = abs(state.closes[-1] - state.closes[0])
        state.er = net / state.sum_abs
    elif len(state.closes) >= 2 and state.sum_abs == 0:
        state.er = 0.0

    state.last_close = c
    return state.er

@dataclass
class FundingState:
    funding_rate: float = np.nan
    premium: float = np.nan
    min_funding_rate: float = np.nan
    max_funding_rate: float = np.nan
    funding_time: Optional[int] = None
    next_funding_time: Optional[int] = None
    
     # 指标：premium 的 EW 均值/方差，用于 z-score
    prem_rv: EWVarState = field(default_factory=lambda: EWVarState(lam=0.94))
    # 平滑的 funding 与 premium
    fr_ema: EMAState = field(default_factory=lambda: EMAState(n=20, alpha=_alpha(20)))
    prem_ema: EMAState = field(default_factory=lambda: EMAState(n=20, alpha=_alpha(20)))

    # 派生量（随更新即时给出）
    premium_z: float = np.nan
    annualized: float = np.nan    # 假设 8 小时一结算：* 3 * 365
    time_to_next_min: float = np.nan

def _annualize_funding_rate(funding_rate: float) -> float:
    return funding_rate * 3 * 365.0

@dataclass
class OIState:
    oi: float = np.nan
    oiCcy: float = np.nan
    oiUsd: float = np.nan
    last_ts: Optional[int] = np.nan

    prev_oi: Optional[float] = np.nan
    d_oi: float = np.nan
    d_oi_rate: float = np.nan
    oi_ema: EMAState = field(default_factory=lambda: EMAState(n=20, alpha=_alpha(20)))



# -------------
#
# -------------
@dataclass
class SeriesState:
    macd: MACDState
    rsi: RSIState
    atr: ATRState
    kdj: KDJState
    rv: EWVarState
    micro: MicroState

    qi: QIMicropriceState
    cvd: CVDState
    kyle: KyleLambdaState
    vpin: VPINState
    squeeze: SqueezeState
    donchian: DonchianState
    er: ERState

    funding: FundingState
    oi: OIState

    last_mid: Optional[float] = np.nan
    prev_close: Optional[float] = np.nan
    ofi_win_ms: int = 5000
    ofi_deq: Deque[Tuple[int, float]] = field(default_factory=deque)

    
@dataclass
class DerivedState:
    """用于缓存‘当前bar的二次/时序摘要’。不存滑窗，滑窗在 FeatureSummarizer 内部维护。"""
    last_ts: Optional[int] = None
    last_tf: Optional[str] = None
    summary: Dict[str, float] = field(default_factory=dict)
    version: str = "v1"         # 可用 summarizer 配置 hash
    dirty: bool = False   

   
class FeatureEnginePD:
    """
    DataFrame 流式处理：
      - update_books(df): 记录最新 spread_bp（由 best_bid/ask 计算）
      - update_trades(df): 维护 5s OFI 累计
      - update_candles(instId, tf, df): 对每根bar更新指标并输出 features DataFrame
    """
    def __init__(self, 
                 ema_fast=12, ema_slow=26, dea_n=9, rsi_n=14, atr_n=14, 
                 kdj_n=9, k_smooth=3, d_smooth=3, ewma_lambda=0.94,
                 kyle_alpha=0.1, vpin_bucket_vol=10_000.0, vpin_window=50,
                 squeeze_bb_k=2.0, squeeze_kc_k=1.5, squeeze_lambda=0.94,
                 donchian_n=20, er_n=10,
                 enable_summary=False, summary_cfg=None
                 ):
        self._cfg = dict(ema_fast=ema_fast, ema_slow=ema_slow, dea_n=dea_n,
                         rsi_n=rsi_n, atr_n=atr_n, kdj_n=kdj_n,
                         k_smooth=k_smooth, d_smooth=d_smooth, ewma_lambda=ewma_lambda,
                         kyle_alpha=kyle_alpha, vpin_bucket_vol=vpin_bucket_vol, 
                         vpin_window=vpin_window, squeeze_bb_k=squeeze_bb_k, 
                         squeeze_kc_k=squeeze_kc_k, squeeze_lambda=squeeze_lambda,
                         donchian_n=donchian_n, er_n=er_n
                         )
        self._series: Dict[Tuple[str,str], SeriesState] = {}  # key: (instId, tf)
        self._shared_tf = "__tick__"
        self.updates = 0
        self.updates_cnt = 0

        self.enable_summary = enable_summary
        scfg = summary_cfg or {"horizons_min": (60,180,420), "slope_alpha": 0.3, "ew_alpha": 0.2}
        self.summarizer = FeatureSummarizer(
            horizons_min=tuple(scfg.get("horizons_min",(60,180,420))),
            slope_alpha=float(scfg.get("slope_alpha",0.3)),
            ew_alpha=float(scfg.get("ew_alpha",0.2)),
        )
        self._derived: Dict[Tuple[str, str], DerivedState] = {}
    

    def _get_shared_state(self, instId: str) -> SeriesState:
        """共享 tick 桶：books/trades/OI/funding 都写这里"""
        return self._get_state(instId, self._shared_tf)

    def _get_state(self, instId: str, tr: str) -> SeriesState:
        key = (instId, tr)
        if key not in self._series:
            self._series[key] = SeriesState(
                macd=macd_init(self._cfg["ema_fast"], self._cfg["ema_slow"], self._cfg["dea_n"]),
                rsi=RSIState(self._cfg["rsi_n"]),
                atr=ATRState(self._cfg["atr_n"]),
                kdj=KDJState(self._cfg["kdj_n"]),
                rv=EWVarState(self._cfg["ewma_lambda"]),
                micro=MicroState(),
                qi=QIMicropriceState(),
                cvd=CVDState(),
                kyle=KyleLambdaState(alpha=self._cfg["kyle_alpha"]),
                vpin=VPINState(bucket_vol=self._cfg["vpin_bucket_vol"],
                               window=self._cfg["vpin_window"]),
                squeeze=SqueezeState(bb_k=self._cfg["squeeze_bb_k"],
                                     kc_k=self._cfg["squeeze_kc_k"],
                                     lam=self._cfg["squeeze_lambda"],
                                     ewvar=EWVarState(lam=self._cfg["squeeze_lambda"])),
                donchian=DonchianState(n=self._cfg["donchian_n"]),
                er=ERState(n=self._cfg["er_n"]),
                funding=FundingState(),
                oi=OIState(),
            )
        return self._series[key]
    
    def _get_derived(self, instId: str, tf: str) -> DerivedState:
        key = (instId, tf)
        if key not in self._derived:
            self._derived[key] = DerivedState(version="v1")
        return self._derived[key]
    
    # ----------- books -> spread_bp -----------
    def update_books(self, df_books: pd.DataFrame, instId: str, tf: str):
        
        if df_books is None or df_books.empty: return
        state = self._get_shared_state(instId)

        for _, r in df_books.iterrows():
            bb, ba = r.get("best_bid"), r.get("best_ask")
            if bb is None or ba is None or bb <= 0.0 or ba <= 0.0:
                continue
            mid = (bb + ba) / 2.0
            spread = ba - bb
            state.micro.spread_bp = (spread / mid * 10000.0) if mid > 0.0 else np.nan
            state.last_mid = mid
            bids, asks = r.get("bids"), r.get("asks")
            
            if not bids and "bids" in r and isinstance(r["bids"], (list, tuple)):
                bids = [(float(p), float(q)) for p, q in r["bids"] if float(p) > 0 and float(q) > 0]
            if not asks and "asks" in r and isinstance(r["asks"], (list, tuple)):
                asks = [(float(p), float(q)) for p, q in r["asks"] if float(p) > 0 and float(q) > 0]
            if bids and asks:
                qi_microprice_update(state.qi, bids, asks)
            self.updates += 1

    # ----------- trades -> ofi_5s -----------
    def update_trades(self, df_trades: pd.DataFrame, instId: str, tf: str):
        if df_trades is None or df_trades.empty: return
        state = self._get_shared_state(instId)

        for _, r in df_trades.iterrows():
            ts = int(r["ts"])
            side = str(r.get("side",""))
            sz = float(r.get("sz",0.0))
            delta = sz if side.lower() == "buy" else (-sz if side.lower() == "sell" else 0.0)
            state.ofi_deq.append((ts, delta))
            
            lo = ts - state.ofi_win_ms
            while state.ofi_deq and state.ofi_deq[0][0] < lo:
                state.ofi_deq.popleft()
            state.micro.ofi_5s = sum(x[1] for x in state.ofi_deq)

            cvd_update(state.cvd, delta)
            vpin_update(state.vpin, delta)
            if state.last_mid is not None:
                kyle_lambda_update(state.kyle, state.last_mid, delta)
            self.updates += 1

    # ----------- candles -> features -----------
    def update_candles(self, df_candles: pd.DataFrame, instId: str, tf: str):
        """
        预期 df_candles 列: ["ts","open","high","low","close", ...]
        返回: 每根 bar 对应一行 features（按 ts 升序）
        注意此处的ts为收盘时间
        """
        if df_candles is None or df_candles.empty: return
        state = self._get_state(instId, tf)
        cloesd_ts = 0
        cloesd = False
        for _, r in df_candles.iterrows():
            ts = int(r["ts"])
            o, h, l, c = map(float, r[["open","high","low","close"]])
            confirm_val = r.get("confirm", 0)
            try:
                is_close = int(confirm_val) == 1
            except Exception:
                is_close = str(confirm_val).lower() in ("1", "true", "t", "yes", "y")
            if not is_close:
                cloesd = True
                cloesd_ts = ts
                continue
            
            dif, dea, hist = macd_update(state.macd, c)

            rsi = rsi_update(state.rsi, c)

            atr = atr_update(state.atr, h, l, c)

            k, d, j = kdj_update(state.kdj, h, l, c, 
                                 k_smooth=self._cfg["k_smooth"], 
                                 d_smooth=self._cfg["d_smooth"])

            if state.prev_close and state.prev_close > 0 and c > 0:
                ret = np.log(c / state.prev_close)
            else:
                ret = 0.0
            rv = ewma_var_update(state.rv, float(ret))
            state.prev_close = c

            er_val = er_update(state.er, c)
            sq_ratio, sq_on = squeeze_update(state.squeeze, state.atr, c)
            up, lo, width, width_norm = donchian_update(state.donchian, h, l, state.atr)

            self.updates += 1
            self.updates_cnt = self.updates
            self.updates = 0

            sum_dict = {}
            if self.enable_summary:
                shared = self._get_shared_state(instId)
                sum_dict = self.summarizer.update_on_bar(
                    instId=instId, tf=tf,
                    close=c,
                    ema_fast=(state.macd.ema_fast.value if state.macd and state.macd.ema_fast else np.nan),
                    ema_slow=(state.macd.ema_slow.value if state.macd and state.macd.ema_slow else np.nan),
                    macd_hist=(state.macd.hist if state.macd else np.nan),
                    rsi=(state.rsi.rsi if state.rsi else np.nan),
                    squeeze_on=(state.squeeze.squeeze_on if state.squeeze else False),
                    spread_bp=(shared.micro.spread_bp if shared and shared.micro else np.nan),
                    ofi_5s=(shared.micro.ofi_5s if shared and shared.micro else np.nan),
                    cvd=(shared.cvd.cvd if shared and shared.cvd else np.nan),
                    kyle_lambda=(shared.kyle.value if shared and shared.kyle else np.nan),
                    vpin=(shared.vpin.vpin if shared and shared.vpin else np.nan),
                    donchian_upper=(state.donchian.upper if state.donchian else np.nan),
                    donchian_lower=(state.donchian.lower if state.donchian else np.nan),
                    atr=(state.atr.atr if state.atr else np.nan),
                    d_oi_rate=(shared.oi.d_oi_rate if shared and shared.oi else np.nan),
                )
                dstate = self._get_derived(instId, tf)
                dstate.last_ts = ts     # 注意：ts 是“收盘时间”
                dstate.last_tf = tf
                dstate.summary = sum_dict or {}
                dstate.dirty = True

            df = self.snapshot_feature(instId, tf, ts)
            return df
        else:
            return None
    
    def update_funding_rate(self, df_funding_rate: pd.DataFrame, instId: str, tf: str):
        """
        预期 df_funding_rate 列字段（兼容不同命名）：
        ts, fundingRate|funding_rate, minFundingRate|min_funding_rate,
        maxFundingRate|max_funding_rate, premium|premium_index,
        fundingTime|funding_time, nextFundingTime|next_funding_time
        允许缺失列；会做稳健处理与 EMA/年化/高级派生计算。
        """
        if df_funding_rate is None or df_funding_rate.empty:
            return
        state = self._get_shared_state(instId)
        frs = state.funding

        for _, r in df_funding_rate.iterrows():
            # 取值（容错多个命名）
            ts = int(r.get("ts", r.get("timestamp", 0)) or 0)
            fr = float(r.get("fundingRate", r.get("funding_rate", np.nan)) or np.nan)
            prem = float(r.get("premium", r.get("premium_index", np.nan)) or np.nan)

            min_fr = float(r.get("minFundingRate", r.get("min_funding_rate", np.nan)) or np.nan)
            max_fr = float(r.get("maxFundingRate", r.get("max_funding_rate", np.nan)) or np.nan)

            ftime = r.get("fundingTime", r.get("funding_time", None))
            nft   = r.get("nextFundingTime", r.get("next_funding_time", None))
            funding_time = int(ftime) if pd.notna(ftime) and ftime is not None else None
            next_funding_time = int(nft) if pd.notna(nft) and nft is not None else None

            # 更新原始
            if pd.notna(fr):
                frs.funding_rate = fr
                ema_update(frs.fr_ema, fr)
            if pd.notna(prem):
                frs.premium = prem
                # prem 的 EW 统计用于 z-score
                _ = ewma_var_update(frs.prem_rv, prem)
                ema_update(frs.prem_ema, prem)
                if frs.prem_rv.mean is not None and frs.prem_rv.var is not None and frs.prem_rv.var >= 0:
                    mu, var = frs.prem_rv.mean, frs.prem_rv.var
                    std = np.sqrt(max(var, 0.0))
                    frs.premium_z = (prem - mu) / std if std > 0 else np.nan

            if pd.notna(min_fr): frs.min_funding_rate = min_fr
            if pd.notna(max_fr): frs.max_funding_rate = max_fr

            frs.funding_time = funding_time
            frs.next_funding_time = next_funding_time

            # 年化（默认 8 小时结算）
            if pd.notna(frs.funding_rate):
                frs.annualized = _annualize_funding_rate(float(frs.funding_rate))

            # 与“当前 bar 收盘 ts”无关时先计算一个到下一次 funding 的剩余分钟（基于该行 ts）
            if next_funding_time is not None and ts:
                dt_ms = max(next_funding_time - ts, 0)
                frs.time_to_next_min = dt_ms / 60000.0
            
            self.updates += 1

    
    def update_open_interest(self, df_oi: pd.DataFrame, instId: str, tf: str):
        """
        预期 df_oi 列字段（兼容不同命名）：
        ts, oi|open_interest, oiCcy|oi_ccy, oiUsd|oi_usd
        会输出 d_oi 与 d_oi_rate（相对变化率），以及对 oi 的 EMA 平滑。
        """
        if df_oi is None or df_oi.empty:
            return
        state = self._get_shared_state(instId)
        ois = state.oi

        for _, r in df_oi.iterrows():
            ts = int(r.get("ts", r.get("timestamp", 0)) or 0)
            oi = r.get("oi", r.get("open_interest", None))
            oiCcy = r.get("oiCcy", r.get("oi_ccy", None))
            oiUsd = r.get("oiUsd", r.get("oi_usd", None))

            oi = float(oi) if oi is not None and pd.notna(oi) else np.nan
            oiCcy = float(oiCcy) if oiCcy is not None and pd.notna(oiCcy) else np.nan
            oiUsd = float(oiUsd) if oiUsd is not None and pd.notna(oiUsd) else np.nan

            # 更新
            if pd.notna(oi):
                ois.prev_oi = ois.oi if pd.notna(ois.oi) else None
                ois.oi = oi
                ema_update(ois.oi_ema, oi)

                if ois.prev_oi is not None and ois.prev_oi > 0:
                    ois.d_oi = oi - ois.prev_oi
                    ois.d_oi_rate = (oi - ois.prev_oi) / ois.prev_oi
                else:
                    ois.d_oi = np.nan
                    ois.d_oi_rate = np.nan

            if pd.notna(oiCcy): ois.oiCcy = oiCcy
            if pd.notna(oiUsd): ois.oiUsd = oiUsd
            ois.last_ts = ts if ts else ois.last_ts

            self.updates += 1

    def snapshot_feature(self, instId: str, tf: str, ts: int, do_round=True):
        state = self._get_state(instId, tf)
        shared = self._series.get((instId, self._shared_tf)) or self._get_shared_state(instId)
        # 共享（与 tf 无关）的来源
        src_micro   = shared.micro
        src_qi      = shared.qi
        src_cvd     = shared.cvd
        src_kyle    = shared.kyle
        src_vpin    = shared.vpin
        src_funding = shared.funding
        src_oi      = shared.oi

        out_rows = []
        out_rows.append({
                "instId": instId, "tf": tf, "ts": ts_to_str(ts),
                "ema_fast": state.macd.ema_fast.value or state.prev_close, "ema_slow": state.macd.ema_slow.value or state.prev_close,
                "macd_dif": state.macd.diff or 0.0, "macd_dea": state.macd.dea.value or 0.0, 
                "macd_hist": state.macd.hist or 0.0, "rsi": state.rsi.rsi or 50.0,
                "kdj_k": state.kdj.k, "kdj_d": state.kdj.d, "kdj_j": state.kdj.j,
                "atr": state.atr.atr or 0.0, "rv_ewma": state.rv.var or 0.0,
                "squeeze_ratio": state.squeeze.ratio, "squeeze_on": state.squeeze.squeeze_on,
                "donchian_upper": state.donchian.upper, "donchian_lower": state.donchian.lower, 
                "donchian_width": state.donchian.width, "donchian_width_norm": state.donchian.width_norm,
                "er": state.er.er,
                
                # From shared
                "spread_bp": src_micro.spread_bp, "ofi_5s": src_micro.ofi_5s,
                "qi1": src_qi.qi1, "qi5": src_qi.qi5, "microprice": src_qi.microprice,
                "cvd": src_cvd.cvd, "kyle_lambda": src_kyle.value, "vpin": src_vpin.vpin,
                "funding_rate": src_funding.funding_rate,
                "funding_rate_ema": src_funding.fr_ema.value,
                "funding_premium": src_funding.premium,
                "funding_premium_ema": src_funding.prem_ema.value,
                "funding_premium_z": src_funding.premium_z,
                "funding_annualized": src_funding.annualized,
                "funding_time": ts_to_str(src_funding.funding_time) if src_funding.funding_time else None,
                "next_funding_time": ts_to_str(src_funding.next_funding_time) if src_funding.next_funding_time else None,
                "funding_time_to_next_min": (
                    max((src_funding.next_funding_time - ts)/60000.0, 0.0)
                    if src_funding.next_funding_time else np.nan
                ),
                "oi": src_oi.oi, "oiCcy": src_oi.oiCcy, "oiUsd": src_oi.oiUsd,
                "d_oi": src_oi.d_oi, "d_oi_rate": src_oi.d_oi_rate,
                "oi_ema": src_oi.oi_ema.value,
            })
        
        # 合并摘要（仅当 ts 匹配；否则忽略，避免跨 bar 污染）
        if self.enable_summary:
            dstate = self._get_derived(instId, tf)
            if dstate.last_ts == ts and dstate.summary:
                out_rows[0].update(dstate.summary)
                dstate.dirty = False   # 被消费过了

        if not out_rows:
            return pd.DataFrame(columns=self.columns())
        
        df = (pd.DataFrame(out_rows)
                .sort_values("ts")
                .drop_duplicates(subset=["ts", "instId", "tf"])
                .reset_index(drop=True))

        if do_round:
            df = round_numeric_columns(df, round_map, do_round=True)
        return df

    def _write_last_summary(self, state, ts: int, summary: dict):
        """把本根bar的 summary 写到 state 上，顺便做基本清洗"""
        if summary is None:
            summary = {}
        clean = {}
        for k, v in summary.items():
            if v is None:
                clean[k] = np.nan
            elif isinstance(v, bool):
                clean[k] = float(v)
            else:
                try:
                    fv = float(v)
                    clean[k] = fv
                except Exception:
                    continue
        state._last_summary = clean
        state._last_summary_ts = int(ts) if ts is not None else None


    def _read_last_summary_for_ts(self, state, ts: int) -> dict:
        """
        只有当缓存的 summary 与当前 ts 一致才返回；否则返回空 dict。
        这样避免把上一根 bar 的摘要误并到本根。
        """
        if getattr(state, "_last_summary_ts", None) == int(ts):
            return getattr(state, "_last_summary", {}) or {}
        return {}

    def columns(self):
        basic_colums = [
            "instId","tf","ts",
            "ema_fast","ema_slow","macd_dif","macd_dea","macd_hist",
            "rsi","kdj_k","kdj_d","kdj_j",
            "atr","rv_ewma","spread_bp","ofi_5s",
            "qi1","qi5","microprice",
            "cvd","kyle_lambda","vpin",
            "squeeze_ratio","squeeze_on",
            "donchian_upper","donchian_lower","donchian_width","donchian_width_norm",
            "er",
            "funding_rate","funding_rate_ema",
            "funding_premium","funding_premium_ema","funding_premium_z",
            "funding_annualized",
            "funding_time","next_funding_time","funding_time_to_next_min",
            "oi","oiCcy","oiUsd","d_oi","d_oi_rate","oi_ema",
        ] 
        if self.enable_summary:
            basic_colums.extend(self.summarizer.summary_columns())
        return basic_colums
