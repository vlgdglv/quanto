# positions.py
from pydantic import BaseModel
from typing import Literal, List, Optional, Dict, Any, Tuple
from pydantic import BaseModel

from agent.schemas import Side

class PositionSnapshot(BaseModel):
    inst: str
    side: Side
    size: float
    entry_price: Optional[float] = None

class TradeIntent(BaseModel):
    side: Side
    action_norm: Literal["ENTER","ADD","REDUCE","REVERSE","SKIP"]
    size_pct: float
    entries: List[float] = []
    stop: float = 0.0
    tps: List[float] = []
    origin: Literal["15m"]
    ts: str
    meta: Dict[str, Any] = {}


def trading_agent_build_intent(*, inst: str, ts: str, rd, timing, pos: PositionSnapshot,
                               default_size_pct: float = 10.0) -> TradeIntent:
    """唯一输出下单意图的地方（最小时间粒度）"""
    if rd.direction == "FLAT" or timing.action in ("SKIP","REDUCE") or not timing.entry_zone or not timing.stop:
        return TradeIntent(side="FLAT", action_norm="SKIP", size_pct=0.0, origin="15m", ts=ts,
                           meta={"reason":"skip or incomplete timing"})

    side: Side = rd.direction
    if pos.side == "FLAT":
        action = "ENTER"
    elif pos.side == side:
        action = "ADD"
    else:
        action = "REVERSE"

    lo, hi = timing.entry_zone
    price = round((lo + hi) / 2, 6)
    return TradeIntent(
        side=side, action_norm=action, size_pct=default_size_pct,
        entries=[price], stop=timing.stop, tps=[*(timing.tps or [])],
        origin="15m", ts=ts, meta={"notes": timing.notes}
    )

class PositionSnapshot(BaseModel):
    inst: str
    side: Side
    size: float
    leverage: float
    entry_price: Optional[float] = None
    pnl_unreal: Optional[float] = None
    equity_free_pct: Optional[float] = None

class PositionProvider:
    async def get_position(self, inst: str) -> PositionSnapshot:
        raise NotImplementedError
