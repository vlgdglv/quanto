# feature/engine_pd.py
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Deque, Tuple, Sequence
import numpy as np
import pandas as pd
from datetime import datetime

from utils.logger import logger

def ts_to_str(ts):
    dt = datetime.fromtimestamp(ts/1000)
    return dt.strftime("%Y%m%d%H%M%S")

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
    return state.diff, dea, state.hist

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

round_map = {
    "rsi": 2, "kdj_k": 2, "kdj_d": 2, "kdj_j": 2,
    "macd_dif": 6, "macd_dea": 6, "macd_hist": 6,
    "atr": 6, "rv_ewma": 6, "er": 3,
    "spread_bp": 2, "qi1": 3, "qi5": 3, "vpin": 3,
    "kyle_lambda": 6, "squeeze_ratio": 3,
    "donchian_width": 6, "donchian_width_norm": 3,
}


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

    last_mid: Optional[float] = None
    prev_close: Optional[float] = None
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
                er=ERState(n=self._cfg["er_n"])
            )
        return self._series[key]
    
    # ----------- books -> spread_bp -----------
    def update_books(self, df_books: pd.DataFrame, instId: str, tf: str):
        if df_books is None or df_books.empty: return
        state = self._get_state(instId, tf)

        for _, r in df_books.iterrows():
            bb, ba = r.get("best_bid"), r.get("best_ask")
            if bb is None or ba is None or bb <= 0.0 or ba <= 0.0:
                continue
            mid = (bb + ba) / 2.0
            spread = ba - bb
            state.micro.spread_bp = (spread / mid * 10000.0) if mid > 0.0 else np.nan
            state.last_mid = mid
            bids, asks = r.get("bids"), r.get("asks")
            # for i in range(1, 6):
            #     bp = bids[i][0]; bq = bids[i][1]
            #     ap = asks[i][0]; aq = asks[i][1]
            #     if bp is not None and bq is not None:
            #         try:
            #             bp_f, bq_f = float(bp), float(bq)
            #             if bp_f > 0 and bq_f > 0:
            #                 bids.append((bp_f, bq_f))
            #         except Exception:
            #             pass
            #     if ap is not None and aq is not None:
            #         try:
            #             ap_f, aq_f = float(ap), float(aq)
            #             if ap_f > 0 and aq_f > 0:
            #                 asks.append((ap_f, aq_f))
            #         except Exception:
            #             pass
            # 2) 或者嵌套列 bids/asks: [(price,size),...]
            if not bids and "bids" in r and isinstance(r["bids"], (list, tuple)):
                bids = [(float(p), float(q)) for p, q in r["bids"] if float(p) > 0 and float(q) > 0]
            if not asks and "asks" in r and isinstance(r["asks"], (list, tuple)):
                asks = [(float(p), float(q)) for p, q in r["asks"] if float(p) > 0 and float(q) > 0]
            if bids and asks:
                qi_microprice_update(state.qi, bids, asks)
                

    # ----------- trades -> ofi_5s -----------
    def update_trades(self, df_trades: pd.DataFrame, instId: str, tf: str):
        if df_trades is None or df_trades.empty: return
        state = self._get_state(instId, tf)

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


    # ----------- candles -> features -----------
    def update_candles(self, df_candles: pd.DataFrame, instId: str, tf: str):
        """
        预期 df_candles 列: ["ts","open","high","low","close", ...]
        返回: 每根 bar 对应一行 features（按 ts 升序）
        注意此处的ts为收盘时间
        """
        if df_candles is None or df_candles.empty: return
        state = self._get_state(instId, tf)
        out_rows = []
        for _, r in df_candles.iterrows():
            ts = int(r["ts"])
            o, h, l, c = map(float, r[["open","high","low","close"]])
            confirm_val = r.get("confirm", 0)
            try:
                is_close = int(confirm_val) == 1
            except Exception:
                is_close = str(confirm_val).lower() in ("1", "true", "t", "yes", "y")

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

            spread_bp = state.micro.spread_bp
            ofi_5s = state.micro.ofi_5s

            qi1 = state.qi.qi1
            qi5 = state.qi.qi5
            microprice = state.qi.microprice
            cvd_val = state.cvd.cvd
            kyle_lambda = state.kyle.value
            vpin_val = state.vpin.vpin

            if not is_close: 
                continue
            out_rows.append({
                "instId": instId, "tf": tf, "ts": ts_to_str(ts),
                "ema_fast": state.macd.ema_fast.value or c,
                "ema_slow": state.macd.ema_slow.value or c,
                "macd_dif": dif or 0.0, "macd_dea": state.macd.dea.value or 0.0, "macd_hist": hist or 0.0,
                "rsi": rsi or 50.0, "kdj_k": k, "kdj_d": d, "kdj_j": j,
                "atr": atr or 0.0, "rv_ewma": rv,
                "spread_bp": spread_bp, "ofi_5s": ofi_5s,
                "qi1": qi1, "qi5": qi5, "microprice": microprice,
                "cvd": cvd_val, "kyle_lambda": kyle_lambda, "vpin": vpin_val,
                "squeeze_ratio": sq_ratio, "squeeze_on": sq_on,
                "donchian_upper": up, "donchian_lower": lo, "donchian_width": width, "donchian_width_norm": width_norm,
                "er": er_val,
            })
        
        if not out_rows:
            return pd.DataFrame(columns=self.columns())
        
        df = (pd.DataFrame(out_rows)
                .sort_values("ts")
                .drop_duplicates(subset=["ts", "instId", "tf"])
                .reset_index(drop=True))
        for col, nd in round_map.items():
            if col in df.columns:
                df[col] = df[col].round(nd)

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
        ] 