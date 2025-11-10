# io_schemas.py 里新增 / 调整
from typing import Optional, List, Tuple, Literal, Dict, Any
from pydantic import BaseModel

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

class TradeIntent(BaseModel):
    side: Side
    size_pct: float
    entries: List[float] = []
    stop: float = 0.0
    tps: List[float] = []
    origin: Literal["1H","30m","15m"]
    ts: str
    meta: Dict[str, Any] = {}
