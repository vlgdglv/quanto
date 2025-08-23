# agent/schema.py
from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, List

Action = Literal["BUY_LONG", "SELL_SHORT", "REDUCE", "CLOSE", "HOLD"]

class ActionProposal(BaseModel):
    instId: str
    tf: str
    ts_decision: int
    action: Action
    # leverage: int 
    target_position: float = Field(..., description="目标仓位占用(0~1)，HOLD可为0")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)

    @field_validator("target_position")
    @classmethod
    def _clip(cls, v):
        return max(min(v, 1.0), 0.0)
    