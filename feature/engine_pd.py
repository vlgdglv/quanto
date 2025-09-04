# feature/engine_pd.py
from __future__ import annotations
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Deque, Tuple, Sequence, List, Any, Callable
import numpy as np
import pandas as pd
from datetime import datetime
import math, time

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
}

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
class _SeriesState:
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
                 donchian_n=20, er_n=10
                 ):
        self._cfg = dict(ema_fast=ema_fast, ema_slow=ema_slow, dea_n=dea_n,
                         rsi_n=rsi_n, atr_n=atr_n, kdj_n=kdj_n,
                         k_smooth=k_smooth, d_smooth=d_smooth, ewma_lambda=ewma_lambda,
                         kyle_alpha=kyle_alpha, vpin_bucket_vol=vpin_bucket_vol, 
                         vpin_window=vpin_window, squeeze_bb_k=squeeze_bb_k, 
                         squeeze_kc_k=squeeze_kc_k, squeeze_lambda=squeeze_lambda,
                         donchian_n=donchian_n, er_n=er_n
                         )
        self._series: Dict[Tuple[str,str], _SeriesState] = {}  # key: (instId, tf)
        self._shared_tf = "__tick__"
        self.updates = 0
        self.updates_cnt = 0

    def _get_shared_state(self, instId: str) -> _SeriesState:
        """共享 tick 桶：books/trades/OI/funding 都写这里"""
        return self._get_state(instId, self._shared_tf)

    def _get_state(self, instId: str, tr: str) -> _SeriesState:
        key = (instId, tr)
        if key not in self._series:
            self._series[key] = _SeriesState(
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
            df = self.snapshot_feature(instId, tf, cloesd_ts)
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
        
        if not out_rows:
            return pd.DataFrame(columns=self.columns())
        
        df = (pd.DataFrame(out_rows)
                .sort_values("ts")
                .drop_duplicates(subset=["ts", "instId", "tf"])
                .reset_index(drop=True))

        if do_round:
            for col, nd in round_map.items():
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").round(nd)
        return df

    @staticmethod
    def columns():
        return [
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
    

# -----------------------
# Snapshot in memory
# -----------------------
class SnapshotStore:
    def __init__(self, keep_n=100):
        self.keep_n = keep_n
        self._base:     Dict[Tuple[str,str], List[Dict[str, Any]]] = defaultdict(list)
        self._derived:  Dict[Tuple[str,str], List[Dict[str, Any]]] = defaultdict(list)

    def append_base(self, row: Dict[str, Any]):
        key = (row["instId"], row["tf"])
        self._base[key].append(dict(row))
        self._gc(key, kind="base")

    def upsert_derived(self, row: Dict[str, Any]):
        key = (row["instId"], row["tf"])
        ts_ms = _to_ts_ms(row.get("ts"))
        row = dict(row)
        row["ts_ms"] = ts_ms
        L = self._derived[key]
        for i in range(len(L)-1, -1, -1):
            if _to_ts_ms(L[i].get("ts_ms", L[i].get("ts"))) == ts_ms:
                L[i] = row
                break
        else:
            L.append(row)
        self._gc(key, kind="derived")
    
    def fetch(self, instId: str, tf: str, start_ms: int, end_ms: int, kind: str)-> pd.DataFrame:
        key = (instId, tf)
        arr = self._base[key] if kind == "base" else self._derived[key]
        rows = []
        for r in arr:
            tms = _to_ts_ms(r.get("ts_ms", r.get("ts")))
            if tms is None: continue
            if start_ms <= tms <= end_ms:
                rows.append(r)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        if "tm_ms" not in df.columns:
            df["tm_ms"] = df["ts"].map(_to_ts_ms)
        return df.sort_values("tm_ms").reset_index(drop=True)
    
    def _gc(self, key: Tuple[str,str], kind: str):
        horizon_ms = self.keep_n * 24 * 3600 * 1000
        now_ms = int(time.time() * 1000)
        arr = self._base[key] if kind == "base" else self._derived[key]
        out = []
        for r in arr:
            tms = _to_ts_ms(r.get("ts_ms", r.get("ts")))
            if tms is None or (now_ms - tms) <= horizon_ms:
                out.append(r)
        if kind == "base":
            self._base[key] = out
        else:
            self._derived[key] = out

class RollingDeque:
    def __init__(self, max_len: Optional[int]=None, horizon_ms: Optional[int]=None):
        self.max_len = max_len
        self.horizon_ms = horizon_ms
        self.deque = deque(maxlen=self.max_len)
    
    def append(self, ts_ms: int, val: float):
        self.deque.append((ts_ms, val))
        self._trim(ts_ms)

    def _trim(self, current_ts_ms: int):
        if self.horizon_ms is not None:
            lo = current_ts_ms - self.horizon_ms
            while self.deque and (self.deque[0][0] is not None) and self.deque[0][0] < lo:
                self.deque.popleft()
        if self.max_len is not None:
            while len(self.deque) > self.max_len:
                self.deque.popleft()
    
    def last_n(self, n: int) -> List[Tuple[int, float]]:
        return list(self.deque)[-n:]
    
    def all(self) -> List[Tuple[int, float]]:
        return list(self.deque)
    

class WindowManager:
    """
    管理 (instId, tf, col) -> RollingDeque
    """
    def __init__(self):
        self._map: Dict[Tuple[str,str,str], RollingDeque] = {}

    def ingest(self, row: Dict[str, Any], columns: List[str], 
                maxlen: Optional[int]=None, horizon_ms: Optional[int]=None):
        instId, tf = row["instId"], row["tf"]
        ts_ms = _to_ts_ms(row.get("ts_ms"), row.get("ts"))
        if ts_ms is None:
            return
        for col in columns:
            val = _safe_float(row.get(col), np.nan)
            if math.isnan(val):
                continue
            key = (instId, tf, col)
            rq = self._map.get(key)
            if rq is None:
                rq = RollingDeque(maxlen=maxlen, horizon_ms=horizon_ms)
                self._map[key] = rq
            rq.append(ts_ms, val)

    def get_last_n(self, instId: str, tf: str, col: str, n: int) -> List[Tuple[int, float]]:
        key = (instId, tf, col)
        return self._map.get(key, RollingDeque()).last_n(n)
    
    def get_time_windows(self, instId: str, tf: str, col: str, end_ts_ms: int, horizon_ms: int) -> List[Tuple[int, float]]:
        rq = self._map.get((instId, tf, col))
        if rq is None:
            return
        arr = rq.all()
        lo = end_ts_ms - horizon_ms
        return [(t, v) for (t, v) in arr if (t is not None) and (lo <= t <= end_ts_ms)]
    
class PctileManager:
    """
    严格滑窗模式：保存窗口内全部样本，计算 percentile_rank 时 O(n log n) 排序（n 为窗口样本数）
    采样近似模式：保存最多 max_samples 个样本（时间窗内），超限后按 FIFO 丢弃，近似分布。
    """
    def __init__(self, mode: str="exact", horizon_days: int=30, max_samples: int=5000):
        assert mode in ("exact", "sample")
        self.mode = mode
        self.horizon_days = horizon_days
        self.horizon_ms = horizon_days * 24 * 3600 * 1000
        self.max_samples = max_samples
        self._map: Dict[Tuple[str,str], deque] = defaultdict(deque)

    def ingest(self, instId: str, feature: str, ts_ms: int, value: float):
        if ts_ms is None or math.isnan(value):
            return
        key = (instId, feature)
        dq = self._map[key]
        dq.append((ts_ms, float(value)))
        lo = ts_ms - self.horizon_ms
        while dq and dq[0][0] < lo:
            dq.popleft()
        if self.mode == "sample":
            while len(dq) > self.max_samples:
                dq.popleft()

    def percentile_rank(self, instId: str, feature: str, ts_ms: int, value: float) -> float:
        key = (instId, feature)
        dq = self._map.get(key)
        if dq is None or len(dq) == 0:
            return np.nan
        lo = ts_ms - self.horizon_ms
        vals = [v for (t, v) in dq if t >= lo and not math.isnan(v)]
        if not vals:
            return np.nan
        vals_sorted = np.sort(np.array(vals, dtype=float))
        rank = np.searchsorted(vals_sorted, value, side="right")
        return rank / float(len(vals_sorted))

class Resampler:
    """
    简易重采样器：从 SnapshotStore(base) 取最近窗口数据，按目标频率聚合。
    以“末值(last)”作为状态类指标的聚合方式（例如 macd_hist）。
    """
    def __init__(self, store: SnapshotStore, base_tf: str="1m"):
        self.store = store
        self.base_tf = base_tf  

    def last_values(self, instId: str, 
                    tf_from: str, column: str, 
                    end_ts_ms: int, target_freq: str, bins: int) -> List[Tuple[int, float]]:
        """
        返回最近 bins 个目标频率桶的 (bucket_ts_ms, last_value) 列表（按时间升序）。
        """
        # 需要的时间范围：bins * freq + 余量 1 桶
        if target_freq.endswith('m'):
            width_min = int(target_freq[:-1])
            width_ms = width_min * 60_000
        elif target_freq.endswith('h'):
            width_hr = int(target_freq[:-1])
            width_ms = width_hr * 3600_000
        else:
            width_ms = 15 * 60_000  # 默认 15m

        need_ms = bins * width_ms + width_ms  # 多取一桶冗余
        start_ms = end_ts_ms - need_ms

        df = self.store.fetch(instId, tf_from, start_ms=start_ms, end_ms=end_ts_ms, kind="base")
        if df.empty or column not in df.columns:
            return []

        # 确保 ts_ms
        if "ts_ms" not in df.columns:
            df["ts_ms"] = df["ts"].map(_to_ts_ms)

        # 计算分桶 key（向下取整到目标 freq）
        df["_bucket"] = df["ts_ms"].map(lambda t: _floor_to_freq(int(t), target_freq))
        # 每桶取最后一个非空值
        grp = df.groupby("_bucket", as_index=False)[[column, "ts_ms"]].last().dropna(subset=[column])
        grp = grp.sort_values("_bucket")
        # 取最近 bins 桶
        if len(grp) > bins:
            grp = grp.iloc[-bins:]
        out = list(zip(grp["_bucket"].astype(int).tolist(), grp[column].astype(float).tolist()))
        return out
    
class RollingOLS:
    """
    维护最多 N 点的 (t, y) 序列，O(1) 更新 OLS 斜率。
    t 可以用分钟数（ts_ms / 60000）。
    """
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.deq: deque = deque()
        self.sum_t = 0.0
        self.sum_y = 0.0
        self.sum_tt = 0.0
        self.sum_ty = 0.0

    def append(self, t: float, y: float):
        self.deq.append((t, y))
        self.sum_t += t
        self.sum_y += y
        self.sum_tt += t*t
        self.sum_ty += t*y
        while len(self.deq) > self.maxlen:
            t0, y0 = self.deq.popleft()
            self.sum_t -= t0
            self.sum_y -= y0
            self.sum_tt -= t0*t0
            self.sum_ty -= t0*y0

    def slope(self) -> float:
        n = len(self.deq)
        if n < 2:
            return np.nan
        denom = n * self.sum_tt - self.sum_t * self.sum_t
        if abs(denom) < 1e-12:
            return np.nan
        return (n * self.sum_ty - self.sum_t * self.sum_y) / denom

# ------------------------------
# Derived feature 插件注册表
# ------------------------------
DerivedFn = Callable[["Context"], Dict[str, Any]]

class DerivedRegistry:
    def __init__(self):
        self._plugins: Dict[str, DerivedFn] = {}

    def register(self, name: str):
        def deco(fn: DerivedFn):
            self._plugins[name] = fn
            return fn
        return deco

    def names(self) -> List[str]:
        return list(self._plugins.keys())
    
    def compute_all(self, ctx: "Context") -> Dict[str, Any]:
        out = {}
        for name, fn in self._plugins.items():
            try:
                part = fn(ctx) or {}
            except Exception as e:
                part = {f"{name}_error": str(e)}
            out.update(part)
        return out


# ------------------------------
# Context：插件的只读视图
# ------------------------------
@dataclass
class Context:
    instId: str
    tf: str
    ts_ms: int
    base_row: Dict[str, Any]
    store: SnapshotStore
    windows: WindowManager
    pctiles: PctileManager
    resampler: Resampler
    config: Dict[str, Any]

    def get_last_n(self, col: str, n: int) -> List[Tuple[int, float]]:
        return self.windows.get_last_n(self.instId, self.tf, col, n)

    def get_time_window(self, col: str, horizon_ms: int) -> List[Tuple[int, float]]:
        return self.windows.get_time_window(self.instId, self.tf, col, self.ts_ms, horizon_ms)

    def resample_last_values(self, col: str, target_freq: str, bins: int) -> List[Tuple[int, float]]:
        return self.resampler.last_values(self.instId, self.tf, col, self.ts_ms, target_freq, bins)

    def percentile_rank(self, feature: str, value: float) -> float:
        return self.pctiles.percentile_rank(self.instId, feature, self.ts_ms, value)

# ------------------------------
# HistoryFeaturizer：主入口
# ------------------------------
class HistoryFeaturizer:
    def __init__(self, 
                 store: SnapshotStore,
                 registry: DerivedRegistry,
                 window_mgr: WindowManager,
                 pctile_mgr: PctileManager,
                 resampler: Resampler,
                 schema_version: str = "derived.v1.0"):
        self.store = store
        self.registry = registry
        self.windows = window_mgr
        self.pctiles = pctile_mgr
        self.resampler = resampler
        self.schema_version = schema_version

        # 为窗口管理声明需要长期维护的列（可按需扩展）
        self._win_cols = [
            "microprice","atr","donchian_lower","donchian_width",
            "macd_hist","cvd","kyle_lambda","vpin"
        ]
        # 分位数管理的列（映射名可与列名一致）
        self._pctile_cols = ["kyle_lambda","vpin"]

    def on_new_base_row(self, base_row: pd.Series | Dict[str, Any])-> Dict[str, Any]:
        """
        在 bar close 时调用。输入：来自 FeatureEnginePD.snapshot_feature 的单行。
        输出：合并后的 {base + derived + 元数据} 字典。
        """
        if isinstance(base_row, pd.Series):
            base_row = base_row.to_dict()
        instId = base_row["instId"]
        tf = base_row["tf"]
        ts_ms = _to_ts_ms(base_row.get("ts_ms", base_row.get("ts")))
        if ts_ms is None:
            # 尝试落盘也行，但这里直接返回原样
            out = dict(base_row)
            out["_schema"] = self.schema_version
            out["_warn"] = "ts_ms_missing"
            return out
        
        # 1) 写入基础快照
        row_to_store = dict(base_row)
        row_to_store["ts_ms"] = ts_ms
        self.store.append_base(row_to_store)

         # 2) 更新窗口列
        self.windows.ingest(row_to_store, columns=self._win_cols, maxlen=1024, horizon_ms=40*24*3600*1000)

        # 3) 更新分位数列
        for col in self._pctile_cols:
            val = _safe_float(base_row.get(col), np.nan)
            if not math.isnan(val):
                self.pctiles.ingest(instId, col, ts_ms, val)

        # 4) 计算派生
        ctx = Context(
            instId=instId, tf=tf, ts_ms=ts_ms, base_row=base_row,
            store=self.store, windows=self.windows, pctiles=self.pctiles,
            resampler=self.resampler, config={"schema": self.schema_version}
        )
        derived = self.registry.compute_all(ctx)

        # 5) 合并并落盘派生
        out = dict(base_row)
        out.update(derived)
        out["ts_ms"] = ts_ms
        out["_schema"] = self.schema_version
        out["_gen_ts_ms"] = int(pd.Timestamp.utcnow().value // 1_000_000)

        self.store.upsert_derived(out)
        return out
    



# ============================================================
#  内置派生特征插件（你可继续添加）
# ============================================================
registry = DerivedRegistry()

@registry.register("donchian_pos")
def _donchian_pos(ctx: Context) -> Dict[str, Any]:
    micro = _safe_float(ctx.base_row.get("microprice"))
    lo = _safe_float(ctx.base_row.get("donchian_lower"))
    width = _safe_float(ctx.base_row.get("donchian_width"))
    eps = 1e-9
    if any(math.isnan(x) for x in [micro, lo, width]):
        return {"donchian_pos": np.nan}
    if width <= 0:
        return {"donchian_pos": np.nan}
    pos = (micro - lo) / max(width, eps)
    return {"donchian_pos": _clip01(pos)}

@registry.register("atr_pct")
def _atr_pct(ctx: Context) -> Dict[str, Any]:
    atr = _safe_float(ctx.base_row.get("atr"))
    micro = _safe_float(ctx.base_row.get("microprice"))
    eps = 1e-9
    if math.isnan(atr) or math.isnan(micro) or micro <= 0:
        return {"atr_pct": np.nan}
    return {"atr_pct": atr / max(micro, eps)}

@registry.register("kyle_lambda_pctile")
def _kyle_pctile(ctx: Context) -> Dict[str, Any]:
    val = _safe_float(ctx.base_row.get("kyle_lambda"))
    if math.isnan(val):
        return {"kyle_lambda_pctile": np.nan, "kyle_lambda_pctile_src": "empty"}
    p = ctx.percentile_rank("kyle_lambda", val)
    return {"kyle_lambda_pctile": _clip01(p), "kyle_lambda_pctile_src": "window_30d"}

@registry.register("vpin_pctile")
def _vpin_pctile(ctx: Context) -> Dict[str, Any]:
    val = _safe_float(ctx.base_row.get("vpin"))
    if math.isnan(val):
        return {"vpin_pctile": np.nan, "vpin_pctile_src": "empty"}
    p = ctx.percentile_rank("vpin", val)
    return {"vpin_pctile": _clip01(p), "vpin_pctile_src": "window_30d"}

@registry.register("macd_hist_slope_15m")
def _macd_hist_slope_15m(ctx: Context) -> Dict[str, Any]:
    """
    从高频基础快照重采样为 15m 桶，对最近 3 桶求 OLS 斜率（单位：每 15m 的变化量）。
    若不足 3 桶，返回 NaN。
    """
    bins = 3
    pairs = ctx.resample_last_values("macd_hist", target_freq="15m", bins=bins)
    if len(pairs) < bins:
        return {"macd_hist_slope_15m": np.nan, "macd_hist_slope_15m_src": "insufficient"}
    # OLS
    ols = RollingOLS(maxlen=bins)
    for t_ms, y in pairs:
        ols.append(_minutes(t_ms), _safe_float(y))
    slope = ols.slope()
    # 也可输出简单差分：
    last_first = _safe_float(pairs[-1][1]) - _safe_float(pairs[0][1])
    return {
        "macd_hist_slope_15m": slope,
        "macd_hist_slope_15m_df": last_first,
        "macd_hist_slope_15m_src": "resample_15m_last"
    }

@registry.register("cvd_slope_30m")
def _cvd_slope_30m(ctx: Context) -> Dict[str, Any]:
    """
    使用当前 tf 的基础序列，在 30 分钟时间窗内对 (t,y) 做 OLS 斜率。
    若样本太少，返回 NaN。
    """
    horizon_ms = 30 * 60_000
    pairs = ctx.get_time_window("cvd", horizon_ms=horizon_ms)
    if len(pairs) < 2:
        return {"cvd_slope_30m": np.nan, "cvd_slope_30m_df": np.nan, "cvd_slope_30m_src": "insufficient"}
    ols = RollingOLS(maxlen=len(pairs))
    for t_ms, y in pairs:
        ols.append(_minutes(t_ms), _safe_float(y))
    slope = ols.slope()  # 单位：每分钟增长量
    # 末前差
    last_first = _safe_float(pairs[-1][1]) - _safe_float(pairs[0][1])
    return {
        "cvd_slope_30m": slope,
        "cvd_slope_30m_df": last_first,
        "cvd_slope_30m_src": "time_window_30m"
    }

def build_history_layer(base_tf_for_resample: str = "1m",
                        pctile_mode: str = "exact") -> HistoryFeaturizer:
    """
    工厂方法：创建一套可用的历史派生层实例。
    pctile_mode: "exact"（严格滑窗）或 "sample"（近似采样）。
    """
    store = SnapshotStore(keep_days=40)
    window_mgr = WindowManager()
    pctile_mgr = PctileManager(mode=pctile_mode, horizon_days=30, max_samples=8000)
    resampler = Resampler(store=store, base_tf=base_tf_for_resample)
    featurizer = HistoryFeaturizer(
        store=store, registry=registry, window_mgr=window_mgr,
        pctile_mgr=pctile_mgr, resampler=resampler, schema_version="derived.v1.0"
    )
    return featurizer