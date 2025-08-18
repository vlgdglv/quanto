# feature/engine_pd.py
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Deque, Tuple
import numpy as np
import pandas as pd

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
        state.avg_gain, state.avg_loss, state.rsi = 0.0, 0.0, 50.0
        return state.rsi
    delta = close - state.last_close
    gain, loss = max(delta, 0.0), min(-delta, 0.0)
    if state.avg_gain is None or state.avg_loss is None:
        state.avg_gain, state.avg_loss = gain, loss
    else:
        n = state.n
        state.avg_gain = (state.avg_gain * (n - 1.0) + gain) / n
        state.avg_loss = (state.avg_loss * (n - 1.0) + loss) / n
    state.last_close = close
    state.rsi = 100.0 if state.avg_loss == 0.0 else (100.0 - 100.0 / (1.0 + state.avg_gain / state.avg_loss) if state.avg_loss > 0.0 else 0.0) 
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
    prev_close: Optional[float] = None
    # trades window in 5s 
    ofi_win_ms: int = 5000
    ofi_deq: Deque[Tuple[int, float]] = field(default_factory=deque)


class FeatureEnginePD:
    """
    DataFrame 流式处理：
      - update_books(df): 记录最新 spread_bp（由 best_bid/ask 计算）
      - update_trades(df): 维护 5s OFI 累计
      - update_candles(instId, tf, df): 对每根bar更新指标并输出 features DataFrame
    """
    def __init__(self, ema_fast=12, ema_slow=26, dea_n=9,
                 rsi_n=14, atr_n=14, kdj_n=9, k_smooth=3, 
                 d_smooth=3, ewma_lambda=0.94):
        self._cfg = dict(ema_fast=ema_fast, ema_slow=ema_slow, dea_n=dea_n,
                         rsi_n=rsi_n, atr_n=atr_n, kdj_n=kdj_n,
                         k_smooth=k_smooth, d_smooth=d_smooth, ewma_lambda=ewma_lambda)
        self._series: Dict[Tuple[str,str], _SeriesState] = {}  # key: (instId, tf)

    def _get_state(self, instId: str, tr: str) -> _SeriesState:
        key = (instId, tr)
        if key not in self._series:
            self._series[key] = _SeriesState(
                macd=macd_init(self._cfg["ema_fast"], self._cfg["ema_slow"], self._cfg["dea_n"]),
                rsi=RSIState(self._cfg["rsi_n"]),
                atr=ATRState(self._cfg["atr_n"]),
                kdj=KDJState(self._cfg["kdj_n"], self._cfg["k_smooth"], self._cfg["d_smooth"]),
                rv=EWVarState(self._cfg["ewma_lambda"]),
                micro=MicroState()
            )
        return self._series[key]
    
    # ----------- books -> spread_bp -----------
    def update_books(self, df_books: pd.DataFrame, instId: str, tr: str):
        if df_books is None or df_books.empty: return
        state = self._get_state(instId, tr)

        for _, r in df_books.iterrows():
            bb, ba = r.get("best_bid"), r.get("best_ask")
            if bb is None or ba is None or bb <= 0.0 or ba <= 0.0:
                continue
            mid = (bb + ba) / 2.0
            spread = ba - bb
            state.micro.spread_bp = (spread / mid * 10000.0) if mid > 0.0 else np.nan

    # ----------- trades -> ofi_5s -----------
    def update_trades(self, df_trades: pd.DataFrame, instId: str, tr: str):
        if df_trades is None or df_trades.empty: return
        state = self._get_state(instId, tr)

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

            dif, dea, hist = macd_update(state.macd, c)

            rsi = rsi_update(state.rsi, c)

            atr = atr_update(state.atr, h, l, c)

            k, d, j = kdj_update(state.kdj, h, l, c)

            if state.prev_close and state.prev_close > 0 and c > 0:
                ret = np.log(c / state.prev_close)
            else:
                ret = 0.0
            rv = ewma_var_update(state.rv, float(ret))
            state.prev_close = c

            spread_bp = state.micro.spread_bp
            ofi_5s = state.micro.ofi_5s

            out_rows.append({
                "instId": instId, "tf": tf, "ts": ts,
                "ema_fast": state.macd.ema_fast.value or c,
                "ema_slow": state.macd.ema_slow.value or c,
                "macd_dif": dif or 0.0, "macd_dea": state.macd.dea.value or 0.0, "macd_hist": hist or 0.0,
                "rsi": rsi or 50.0, "kdj_k": k, "kdj_d": d, "kdj_j": j,
                "atr": atr or 0.0, "rv_ewma": rv,
                "spread_bp": spread_bp, "ofi_5s": ofi_5s,
            })

        return (pd.DataFrame(out_rows)
                .sort_values("ts")
                .drop_duplicates(subset=["ts", "instId", "tf"])
                .reset_index(drop=True))
    
    @staticmethod
    def columns():
        return [
            "instId","tf","ts",
            "ema_fast","ema_slow","macd_dif","macd_dea","macd_hist",
            "rsi","kdj_k","kdj_d","kdj_j",
            "atr","rv_ewma","spread_bp","ofi_5s"
        ] 