# trading/services/instrument_service.py
from __future__ import annotations
import asyncio, re, math
from typing import Dict, Optional, Any, Union, Sequence
from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_HALF_UP
from trading.models import Instrument
from trading.errors import PrecisionError


def _f(x: Any) -> float:
    """Safe float from str/float/int."""
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0

def _nearest_multiple(value: float, step: float) -> float:
    if step <= 0:
        return value
    q = round(value / step)
    return q * step


def _floor_multiple(value: float, step: float) -> float:
    if step <= 0:
        return value
    q = math.floor(value / step)
    return q * step


def _is_multiple(value: float, step: float, *, eps: float = 1e-8) -> bool:
    if step <= 0:
        return True
    k = round(value / step)
    return abs(value - k * step) <= eps * max(1.0, step)


class InstrumentService:
    """
    Pulls & caches instrument specs (tickSz/lotSz/minSz/ctVal),
    and provides normalization/validation helpers.
    """
    def __init__(self, http_client, endpoints, *, default_inst_type: str = "SPOT", init_cache_instruments: list[str] = []) -> None:
        self._http = http_client
        self._ep = endpoints
        self._cache: Dict[str, Instrument] = {}
        self._lock = asyncio.Lock()
        self._default_inst_type = default_inst_type
        

    @staticmethod
    def _inst_type_from_inst_id(instId: str) -> Optional[str]:
        """
        根据 OKX instId 推断交易品种类型。

        - BTC-USDT-SWAP       -> SWAP
        - BTC-USDT            -> SPOT
        - BTC-USD-230915      -> FUTURES
        - BTC-USD-230915-40000-C -> OPTION
        """
        if not instId or "-" not in instId:
            return None

        parts = instId.split("-")
        last = parts[-1].upper()

        if len(parts) == 2:
            return "SPOT"
        if last == "SWAP":
            return "SWAP"
        if re.fullmatch(r"\d{6}", last):
            return "FUTURES"
        if len(parts) >= 5 and last in ("C", "P"):
            return "OPTION"

        # other cases
        return None
    
    async def refresh(self, 
                      instType: Optional[str] = None,
                      instId: Optional[str] = None,
                      prune: bool = False,
                      ) -> None:
        """Fetch /public/instruments?instType=SWAP and populate cache."""
        instType = (instType or self._default_inst_type).upper()

        params = {"instType": instType}
        if instId:
            params["instId"] = instId
            _type_from_id = self._inst_type_from_inst_id(instId)
            if _type_from_id and _type_from_id != instType:
                if getattr(self._http, "log", None):
                    self._http.log.warning(
                        f"Inferred instType {_type_from_id} from instId {instId}, overridden with {instType}"
                    )
                instType = _type_from_id
            params["instType"] = instType
        
        payload = await self._http.get_public(self._ep.public_instruments, params=params)

        data = payload.get("data") or []
        
        updates: Dict[str, Instrument] = {}
        for row in data:
            inst_id_for_log = row.get("instId", "?")
            try:
                inst_id = row["instId"]
                tick_sz = _f(row["tickSz"])
                lot_sz  = _f(row["lotSz"])
                min_sz  = _f(row.get("minSz", "0"))
                ct_val  = _f(row.get("ctVal", "0"))

                inst = Instrument(
                    instId=inst_id,
                    tickSz=tick_sz,
                    lotSz=lot_sz,
                    minSz=min_sz,
                    ctVal=ct_val,
                )
                updates[inst_id] = inst
            except Exception as e:
                if getattr(self._http, "log", None):
                    self._http.log.warning(f"Failed to parse instrument {inst_id_for_log}: row: {row} (err={e})")
        
        async with self._lock:
            if not prune:
                merged = dict(self._cache)
                merged.update(updates)
                self._cache = merged
                return
            
            if not instId:
                self._cache = updates
                return
            else:
                new_cache = dict(self._cache)
                new_cache.pop(instId, None)
                new_cache.update(updates)
                self._cache = new_cache

    async def get_or_refresh(self, instId: str) -> Instrument:
        inst = self._cache.get(instId)
        if inst:
            return inst
        
        instType = self._inst_type_from_inst_id(instId) or self._default_inst_type
        await self.refresh(instType, instId)
        inst = self._cache.get(instId)
        if not inst:
            raise KeyError(f"Unknown instrument after refresh: {instId}")
        return inst
    
    def get(self, instId: str) -> Instrument:
        """Return instrument specs or raise if unknown."""
        inst = self._cache.get(instId)
        if not inst:
            raise KeyError(f"Unknown instrument: {instId}")
        return inst

    def round_price(self, instId: str, px: float) -> float:
        """Round price to tickSz."""
        inst = self.get(instId)
        step = inst.tickSz
        rounded = _nearest_multiple(float(px), step)
        return rounded

    def normalize_size(self, instId: str, sz: float) -> float:
        """Round size to lotSz multiples."""
        inst = self.get(instId)
        step = inst.lotSz
        norm = _floor_multiple(float(sz), step)
        return norm

    def validate(self, instId: str, px: float | None, sz: float) -> None:
        """Validate minSz and precision constraints; raise PrecisionError on violation."""
        inst = self.get(instId)
        lot    = float(inst.lotSz)
        tick   = float(inst.tickSz)
        min_sz = float(inst.minSz)
        sz_f   = float(sz)

        if sz_f < min_sz:
            raise PrecisionError(
                f"Size {sz_f} < minSz {min_sz} for {instId}",
                field="sz", expected=f">={min_sz}", actual=str(sz_f), instId=instId
            )

        if not _is_multiple(sz_f, lot):
            fixed = _floor_multiple(sz_f, lot)
            raise PrecisionError(
                f"Size {sz_f} not multiple of lotSz {lot} for {instId}",
                field="sz", expected=f"multiple of {lot}", actual=str(sz_f),
                suggestion=fixed, instId=instId
            )

        if px is not None:
            px_f = float(px)
            if not _is_multiple(px_f, tick):
                fixed_px = _nearest_multiple(px_f, tick)
                raise PrecisionError(
                    f"Price {px_f} not multiple of tickSz {tick} for {instId}",
                    field="px", expected=f"multiple of {tick}", actual=str(px_f),
                    suggestion=fixed_px, instId=instId
                )