# feature/summarizer.py
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Deque, Tuple, Optional, List
import math
import numpy as np

@dataclass
class EWMean:
    alpha: float
    value: Optional[float] = None
    def update(self, x: float) -> float:
        if x is None or np.isnan(x): 
            return self.value if self.value is not None else np.nan
        self.value = x if self.value is None else (self.alpha * x + (1 - self.alpha) * self.value)
        return self.value
    
@dataclass
class EWWelford:
    alpha: float
    mean: Optional[float] = None
    var: Optional[float] = None
    def update(self, x: float):
        if x is None or np.isnan(x): 
            return self.mean, self.var
        if self.mean is None:
            self.mean, self.var = x, 0.0
            return self.mean, self.var
        prev_mean = self.mean
        self.mean = self.alpha * x + (1 - self.alpha) * self.mean
        # GARCH 风格的 EW 方差（稳定、O(1)）
        self.var = self.alpha * (x - self.mean) ** 2 + (1 - self.alpha) * (self.var if self.var is not None else 0.0)
        return self.mean, self.var
    
@dataclass
class RollingDeque:
    maxlen: int
    dq: Deque[float] = field(default_factory=deque)
    s: float = 0.0
    s2: float = 0.0

    def push(self, x: float):
        x = float(x) if x is not None and not np.isnan(x) else 0.0
        self.dq.append(x)
        self.s += x
        self.s2 += x * x
        if len(self.dq) > self.maxlen:
            y = self.dq.popleft()
            self.s -= y
            self.s2 -= y * y

    def mean(self) -> float:
        n = len(self.dq)
        return self.s / n if n > 0 else np.nan
    
    def std(self) -> float:
        n = len(self.dq)
        if n<=1: return np.nan
        m = self.s / n
        v = max(self.s2 / n - m * m, 0.0)
        return math.sqrt(v)
    
    def sum(self) -> float:
        return self.s
    
    def last(self) -> float:
        return self.dq[-1] if len(self.dq) > 0 else np.nan

@dataclass
class RollingOLSSlope:
    """固定窗口的最小二乘直线斜率，O(1) 更新"""
    maxlen: int
    dq: deque = field(default_factory=deque)  # 保存 (x, y)
    n: int = 0
    Sx: float = 0.0
    Sy: float = 0.0
    Sxx: float = 0.0
    Sxy: float = 0.0

    def push(self, x: float, y: float):
        # 入队
        self.dq.append((x, y))
        self.n += 1
        self.Sx += x; self.Sy += y
        self.Sxx += x*x; self.Sxy += x*y
        # 出队
        if self.n > self.maxlen:
            x0, y0 = self.dq.popleft()
            self.n -= 1
            self.Sx -= x0; self.Sy -= y0
            self.Sxx -= x0*x0; self.Sxy -= x0*y0

    def slope(self) -> float:
        if self.n < 2:
            return np.nan
        denom = self.n*self.Sxx - self.Sx*self.Sx
        if abs(denom) < 1e-12:
            return 0.0
        return (self.n*self.Sxy - self.Sx*self.Sy) / denom
    
@dataclass
class StreakCounter:
    last_sign: int = 0
    len: int = 0
    def update(self, x: float):
        sign = 1 if x > 0 else (-1 if x < 0 else 0)
        if sign == self.last_sign and sign != 0:
            self.len += 1
        elif sign != 0:
            self.len = 1
        else:
            self.len = 0
        self.last_sign = sign
        return self.len

@dataclass
class BoolDuration:
    last: Optional[bool] = None
    dur: int = 0
    def update(self, flag: bool):
        if flag is None:
            return 0
        if self.last is None:
            self.last = bool(flag)
            self.dur = 1 if self.last else 0
        else:
            if bool(flag) == self.last:
                self.dur += 1 if self.last else 0
            else:
                self.last = bool(flag)
                self.dur = 1 if self.last else 0
        return self.dur
    
@dataclass
class SummaryState:
    """按 tf 维护，不同 horizon 各有一套滑窗"""
    # 窗口：horizon 步长（bar 数）在构造时设定
    win_mom: Dict[str, RollingDeque] = field(default_factory=dict)     # 动量：价差/ema_spread/macd_hist
    win_rsi: Dict[str, RollingDeque] = field(default_factory=dict)     # rsi
    win_spread: Dict[str, RollingDeque] = field(default_factory=dict)  # 点差
    win_ofi: Dict[str, RollingDeque] = field(default_factory=dict)     # ofi 汇总
    win_cvd: Dict[str, RollingDeque] = field(default_factory=dict)     # cvd 增量
    win_kyle: Dict[str, EWMean] = field(default_factory=dict)          # kyle 低频平滑
    win_vpin: Dict[str, EWMean] = field(default_factory=dict)          # vpin 平滑
    win_oi_rate: Dict[str, RollingDeque] = field(default_factory=dict) # oi 变化率
    # 斜率/趋势：用 EMA 平滑的差分近似 slope
    slope_mom: Dict[str, EWMean] = field(default_factory=dict)
    slope_rsi: Dict[str, EWMean] = field(default_factory=dict)
    # 连续性/状态时长
    streak_macd_pos: StreakCounter = field(default_factory=StreakCounter)
    streak_macd_neg: StreakCounter = field(default_factory=StreakCounter)
    regime_squeeze_on: BoolDuration = field(default_factory=BoolDuration)


class FeatureSummarizer:
    """
    维护“LLM 友好”的时序摘要，O(1) 更新，按 horizon 输出：
      - s_mom_slope_{Hx}, s_rsi_mean_{Hx}, s_rsi_std_{Hx}
      - s_macd_pos_streak, s_macd_neg_streak
      - s_ofi_sum_30m, s_cvd_delta_{Hx}, s_spread_bp_med_{Hx}（用 mean 近似 med）
      - s_kyle_ema_{Hx}, s_vpin_mean_{Hx}
      - s_squeeze_on_dur
      - s_donchian_dist_upper/lower（当下与轨道距离的 z 风格标准化）
      - s_oi_rate_{Hx}
    """
    def __init__(self, horizons_min=(60, 180, 420), slope_alpha=0.3, ew_alpha=0.2):
        self.horizons_min = tuple(horizons_min)  # 1h/3h/7h 等
        self.slope_alpha = slope_alpha
        self.ew_alpha = ew_alpha
        self._states: Dict[Tuple[str, str], SummaryState] = {}

    @staticmethod
    def _tf_to_minutes(tf: str) -> int:
        # 仅处理常见 tf，比如 "1m","5m","15m","1h"
        tf = tf.strip().lower()
        if tf.endswith("m"):
            return int(tf[:-1])
        if tf.endswith("h"):
            return int(tf[:-1]) * 60
        raise ValueError(f"unsupported tf: {tf}")

    def _ensure_state(self, instId: str, tf: str) -> SummaryState:
        key = (instId, tf)
        if key in self._states:
            return self._states[key]
        minutes = self._tf_to_minutes(tf)
        st = SummaryState()
        for H in self.horizons_min:
            steps = max(int(H // minutes), 1)
            tag = f"H{H}m"
            st.win_mom[tag] = RollingDeque(steps)
            st.win_rsi[tag] = RollingDeque(steps)
            st.win_spread[tag] = RollingDeque(steps)
            st.win_ofi[tag] = RollingDeque(max(int(30 // minutes), 1))  # 固定 30m 汇总窗口
            st.win_cvd[tag] = RollingDeque(steps)
            st.win_kyle[tag] = EWMean(self.ew_alpha)
            st.win_vpin[tag] = EWMean(self.ew_alpha)
            st.win_oi_rate[tag] = RollingDeque(steps)
            st.slope_mom[tag] = EWMean(self.slope_alpha)
            st.slope_rsi[tag] = EWMean(self.slope_alpha)
        self._states[key] = st
        return st

    def update_on_bar(self, instId: str, tf: str, *,
                       close: float,
                       ema_fast: float, ema_slow: float,
                       macd_hist: float, rsi: float,
                       squeeze_on: bool, spread_bp: float,
                       ofi_5s: float, cvd: float,
                       kyle_lambda: float, vpin: float,
                       donchian_upper: float, donchian_lower: float, atr: float,
                       d_oi_rate: float) -> Dict[str, float]:
        st = self._ensure_state(instId, tf)
        # —— 基础派生
        ema_spread = (ema_fast - ema_slow) if (ema_fast is not None and ema_slow is not None) else np.nan
        # macd 连续性（两种 streak 分开算）
        macd_pos_len = st.streak_macd_pos.update(macd_hist if macd_hist is not None else 0.0) if macd_hist and macd_hist > 0 else (st.streak_macd_pos.update(0.0))
        macd_neg_len = st.streak_macd_neg.update(-macd_hist if macd_hist is not None else 0.0) if macd_hist and macd_hist < 0 else (st.streak_macd_neg.update(0.0))
        # 布尔 regime 时长
        sq_dur = st.regime_squeeze_on.update(bool(squeeze_on) if squeeze_on is not None else False)

        # —— 逐 horizon 更新
        out: Dict[str, float] = {}
        for H in self.horizons_min:
            tag = f"H{H}m"

            # 动量窗口（用 ema_spread 或 macd_hist 皆可；这里取 ema_spread 更稳定）
            if not np.isnan(ema_spread):
                st.win_mom[tag].push(ema_spread)
                # slope 近似：当前-均值 的 EW 平滑
                mom_slope = st.slope_mom[tag].update(ema_spread - st.win_mom[tag].mean())
                out[f"s_mom_slope_{tag}"] = mom_slope

            # rsi 窗口
            if rsi is not None and not np.isnan(rsi):
                st.win_rsi[tag].push(rsi)
                out[f"s_rsi_mean_{tag}"] = st.win_rsi[tag].mean()
                out[f"s_rsi_std_{tag}"]  = st.win_rsi[tag].std()

            # 点差
            if spread_bp is not None and not np.isnan(spread_bp):
                st.win_spread[tag].push(spread_bp)
                out[f"s_spread_bp_mean_{tag}"] = st.win_spread[tag].mean()

            # ofi：统一用 30m 汇总（窗口初始化时已固定 30m）
            if ofi_5s is not None and not np.isnan(ofi_5s):
                st.win_ofi[tag].push(ofi_5s)  # 注意：传的是 bar 末 5s ofi 的“代表值”；如果有 bar 内累加，换成累计
            out["s_ofi_sum_30m"] = st.win_ofi[tag].sum()

            # cvd 增量（相对窗口首末）
            if cvd is not None and not np.isnan(cvd):
                st.win_cvd[tag].push(cvd)
                # 用末值-均值近似窗口增量
                out[f"s_cvd_delta_{tag}"] = (st.win_cvd[tag].last() - st.win_cvd[tag].mean())

            # kyle/vpin 平滑
            if kyle_lambda is not None and not np.isnan(kyle_lambda):
                out[f"s_kyle_ema_{tag}"] = st.win_kyle[tag].update(kyle_lambda)
            if vpin is not None and not np.isnan(vpin):
                out[f"s_vpin_mean_{tag}"] = st.win_vpin[tag].update(vpin)

            # OI 变化率
            if d_oi_rate is not None and not np.isnan(d_oi_rate):
                st.win_oi_rate[tag].push(d_oi_rate)
                out[f"s_oi_rate_{tag}"] = st.win_oi_rate[tag].mean()

        # 非 horizon 相关的摘要
        out["s_macd_pos_streak"] = float(macd_pos_len or 0)
        out["s_macd_neg_streak"] = float(macd_neg_len or 0)
        out["s_squeeze_on_dur"]  = float(sq_dur or 0)

        # donchian 距离：用 ATR 标准化
        if all(v is not None and not np.isnan(v) for v in (donchian_upper, donchian_lower, close, atr)) and atr > 0:
            mid = 0.5 * (donchian_upper + donchian_lower)
            out["s_donchian_dist_upper"] = (donchian_upper - close) / atr
            out["s_donchian_dist_lower"] = (close - donchian_lower) / atr
            out["s_donchian_mid_dev"]    = (close - mid) / atr

        return out

    @staticmethod
    def summary_columns(horizons_min=(60,180,420)) -> List[str]:
        cols = [
            "s_macd_pos_streak","s_macd_neg_streak","s_squeeze_on_dur",
            "s_donchian_dist_upper","s_donchian_dist_lower","s_donchian_mid_dev",
            "s_ofi_sum_30m",
        ]
        for H in horizons_min:
            tag = f"H{H}m"
            cols += [
                f"s_mom_slope_{tag}",
                f"s_rsi_mean_{tag}", f"s_rsi_std_{tag}",
                f"s_spread_bp_mean_{tag}",
                f"s_cvd_delta_{tag}",
                f"s_kyle_ema_{tag}", f"s_vpin_mean_{tag}",
                f"s_oi_rate_{tag}",
            ]
        return cols