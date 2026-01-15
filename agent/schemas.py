# schemas.py
from typing import Optional, List, Tuple, Literal, Dict, Any
from pydantic import BaseModel
from dataclasses import dataclass, field
from enum import Enum, auto
import time

TF = Literal["1m","5m","15m","30m","4H","1H"]
Side = Literal["LONG","SHORT","FLAT"]

class FeatureFrame(BaseModel):
    inst: str
    tf: TF
    ts_close: str
    features: Dict[str, Any]
    kind: Optional[str]

class RegimeSignal(BaseModel):
    regime: Side
    confidence: float
    rationale: List[str] = []
    ts: str

class DirectionSignal(BaseModel):
    side: Side
    conviction: float
    reasons: List[str] = []
    ts: str

class TimingSignal(BaseModel):
    action: Literal["ENTER","ADD","REDUCE","SKIP"]
    entry_zone: Optional[Tuple[float,float]] = None
    stop: Optional[float] = None
    tps: Optional[List[float]] = None
    notes: List[str] = []
    ts: str


@dataclass
class TradePlan:
    inst: str
    side: Side
    size: float           # 合约张数或名义价值（你自己定义）
    leverage: int
    entry_price: float    # 当前参考价（计划的“中值”）
    stop_price: float     # 硬止损
    tp_price: float       # 第一止盈
    created_ts: float

    # 时间相关（MVP：可写死 horizon / min_hold）
    horizon_sec: int      # 期望最大持仓时间（超过则 time-stop）
    min_hold_sec: int     # 最短持仓时间（避免刚进就被噪声震出）

    # 可选的 alpha 元信息（方便追溯）
    alpha_id: Optional[str] = None
    alpha_payload: Optional[Dict[str, Any]] = None  # 原始 alpha JSON
    note: str = ""         # 人类可读摘要（可以由 LLM 写）

    def is_long(self) -> bool:
        return self.side == "LONG"

    def is_short(self) -> bool:
        return self.side == "SHORT"
    
    
