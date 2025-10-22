# agent/schema.py
from pydantic import BaseModel, Field, field_validator
from typing import Literal, List
from enum import Enum

Action = Literal["BUY_LONG", "SELL_SHORT", "REDUCE", "CLOSE", "HOLD"]

class Direction(str, Enum):
    BUY_LONG = "BUY_LONG"
    SELL_SHORT = "SELL_SHORT"
    HOLD = "HOLD"


class ActionProposal(BaseModel):
    instId: str
    tf: str
    ts_decision: int
    action: Action
    leverage: int 
    target_position: float = Field(..., description="目标仓位占用(0~1)，HOLD可为0")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)

    @field_validator("target_position")
    @classmethod
    def _clip(cls, v):
        return max(min(v, 1.0), 0.0)
    
class VolBucket(str, Enum):
    LOW = "LOW"; MID = "MID"; HIGH = "HIGH"

class RRFOut(BaseModel):
    gate: Literal["OPEN", "SOFT", "HARD_VETO"] = "SOFT"
    vol_bucket: VolBucket
    risk_budget: float = Field(..., ge=0, le=1)        # 尺寸上限
    budget_floor: float = Field(0.0, ge=0, le=1)       # 尺寸地板（建议 0~0.1）
    leverage_cap: int = Field(..., ge=0)
    soft_flags: List[str] = []                         # e.g. ["near_funding", "wide_spread"]
    veto_reasons: List[str] = []                       # 仅 HARD_VETO 填写

class SignalScores(BaseModel):
    trend: float
    flow: float
    composite: float

class DDSOut(BaseModel):
    direction: Direction
    base_position: float = Field(..., ge=0, le=1)
    signal_scores: SignalScores
    confidence: float = Field(..., ge=0, le=1)
    rationale: List[str] = []

class EntryCfg(BaseModel):
    mode: str = "LIMIT"
    limit_slip_bp: int = 3

class StopCfg(BaseModel):
    type: str = "ATR_MULT"; mult: float = 1.2

class TakeProfitCfg(BaseModel):
    rr_min: float = 1.6; trail_atr_mult: float = 0.8

class AutoReduceCfg(BaseModel):
    conf_drop_pct: int = 40; flow_flip: bool = True

class RiskControls(BaseModel):
    entry: EntryCfg = EntryCfg()
    stop: StopCfg = StopCfg()
    take_profit: TakeProfitCfg = TakeProfitCfg()
    auto_reduce: AutoReduceCfg = AutoReduceCfg()

class EPMOut(BaseModel):
    action: Direction
    target_position: float = Field(..., ge=0, le=1)
    leverage: int = Field(..., ge=1)
    risk_controls: RiskControls = RiskControls()
    notes: List[str] = []