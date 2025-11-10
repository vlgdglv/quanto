# shared_state.py
import asyncio, time
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta, timezone
from agent.schemas import RegimeSignal, DirectionSignal
from typing import Optional, Literal, List

from agent.schemas import Side

@dataclass
class RDState:
    regime: Side
    regime_confidence: float
    direction: Side
    direction_confidence: float
    invalidation: List[str]
    updated_at: float
    valid_until: float
    version: int

class SharedState:
    def __init__(self, rd_ttl_sec: int = 3600 * 2):
        self._rd: Optional[RDState] = None
        self._rd_ver = 0
        self._rd_ttl = rd_ttl_sec
        self._cond = asyncio.Condition()

    def _now(self) -> float: return time.time()

    async def set_rd(self, *, regime: Side, regime_conf: float, direction: Side, dir_conf: float, invalidation: List[str]):
        async with self._cond:
            self._rd_ver += 1
            now = self._now()
            self._rd = RDState(regime, regime_conf, direction, dir_conf, invalidation,
                               updated_at=now, valid_until=now + self._rd_ttl, version=self._rd_ver)
            self._cond.notify_all()

    async def set_rd(self, *, regime: Side, regime_conf: float, direction: Side, dir_conf: float, invalidation: List[str]):
        await self.merge_rd(regime=regime, regime_conf=regime_conf,
                            direction=direction, dir_conf=dir_conf,
                            invalidation=invalidation)

    async def merge_rd(self, *,
                       regime: Optional[Side] = None,
                       regime_conf: Optional[float] = None,
                       direction: Optional[Side] = None,
                       dir_conf: Optional[float] = None,
                       invalidation: Optional[List[str]] = None):
        async with self._cond:
            now = self._now()
            if self._rd is None:
                base_regime = regime if regime is not None else "None"
                base_regime_conf = regime_conf if regime_conf is not None else 0.0
                base_dir = direction if direction is not None else "None"
                base_dir_conf = dir_conf if dir_conf is not None else 0.0
                base_inv = invalidation if invalidation is not None else []
                self._rd = RDState(base_regime, base_regime_conf,
                                   base_dir, base_dir_conf,
                                   base_inv, updated_at=now,
                                   valid_until=now + self._rd_ttl,
                                   version=1)
                self._rd_ver = 1
            else:
                r = regime if regime is not None else self._rd.regime
                rc = regime_conf if regime_conf is not None else self._rd.regime_confidence
                d = direction if direction is not None else self._rd.direction
                dc = dir_conf if dir_conf is not None else self._rd.direction_confidence
                inv = (invalidation if invalidation is not None else self._rd.invalidation) or []
                self._rd_ver += 1
                self._rd = RDState(r, rc, d, dc, inv,
                                   updated_at=now,
                                   valid_until=now + self._rd_ttl,
                                   version=self._rd_ver)
            self._cond.notify_all()

    def get_rd(self) -> Optional[RDState]:
        return self._rd

    def fresh_rd(self) -> bool:
        return self._rd is not None and self._rd.valid_until > self._now()
    
    def not_ready_reason(self) -> Optional[str]:
        if self._rd is None:
            return "RD is None"
        now = self._now()
        if self._rd.valid_until <= now:
            return f"RD expired at {self._rd.valid_until:.0f} (now={now:.0f})"
        return "Unknown reason"

    # async def wait_rd(self, timeout_sec: float = 2.0) -> bool:
    #     async with self._cond:
    #         return await self._cond.wait_for(self.fresh_rd, timeout=timeout_sec)
    async def wait_rd(self, timeout_sec: float = 2.0) -> bool:
        async with self._cond:
            try:
                # asyncio.Condition.wait_for(predicate) 本身不支持 timeout
                return await asyncio.wait_for(
                    self._cond.wait_for(lambda: self.fresh_rd()),
                    timeout=timeout_sec
                )
            except asyncio.TimeoutError:
                return False