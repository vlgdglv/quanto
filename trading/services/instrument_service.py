# trading/services/instrument_service.py
from __future__ import annotations
import asyncio, re
from typing import Dict, Optional, Any, Union, Sequence
from decimal import Decimal, getcontext, ROUND_DOWN, ROUND_HALF_UP
from trading.models import Instrument
from trading.errors import PrecisionError

# getcontext().prec = 28

def _D(x: Any) -> Decimal:
    """Safe Decimal from str/float/int; prefer str to keep exact tick like '0.0005'."""
    if x is None:
        return Decimal(0)
    if isinstance(x, Decimal):
        return x
    if isinstance(x, int):
        return Decimal(x)
    s = str(x).strip()
    if s == "":
        return Decimal(0)
    return Decimal(s)

def _nearest_multiple(value: Decimal, step: Decimal) -> Decimal:
    """四舍五入到最近的 step 整数倍（用于价格）。"""
    if step <= 0:
        return value
    q = (value / step).to_integral_value(rounding=ROUND_HALF_UP)
    return q * step

def _floor_multiple(value: Decimal, step: Decimal) -> Decimal:
    """向下取到最近的 step 整数倍（用于数量）。"""
    if step <= 0:
        return value
    q = (value / step).to_integral_value(rounding=ROUND_DOWN)
    return q * step

def _is_multiple(value: Decimal, step: Decimal) -> bool:
    if step <= 0:
        return True
    quantized = _nearest_multiple(value, step)
    return quantized == value

class InstrumentService:
    """
    Pulls & caches instrument specs (tickSz/lotSz/minSz/ctVal),
    and provides normalization/validation helpers.
    """

    def __init__(self, http_client, endpoints, *, default_inst_type: str = "SPOT") -> None:
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
                tick_sz = _D(row["tickSz"])
                lot_sz  = _D(row["lotSz"])
                min_sz  = _D(row.get("minSz", "0"))
                ct_val  = _D(row.get("ctVal", "0"))

                inst = Instrument(
                    instId=inst_id,
                    tickSz=tick_sz,
                    lotSz=lot_sz,
                    minSz=min_sz,
                    ctVal=ct_val,
                )
                # new_cache[inst_id] = inst
                updates[inst_id] = inst
            except Exception as e:
                # 忽略单条异常，但建议打日志
                # 例如某些奇葩产品返回字段为空/非法
                # logger 可用 self._http.log 或上层注入
                # self._http.log.warning("Failed to parse instrument %s: %s", inst_id, e)
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
        """优雅兜底：若缓存没有该 inst，自动 refresh 一次再取。"""
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
        px_d = _D(px)
        step = _D(inst.tickSz)
        rounded = _nearest_multiple(px_d, step)
        return float(rounded)

    def normalize_size(self, instId: str, sz: float) -> float:
        """Round size to lotSz multiples."""
        inst = self.get(instId)
        sz_d = _D(sz)
        step = _D(inst.lotSz)
        norm = _floor_multiple(sz_d, step)
        return float(norm)

    def validate(self, instId: str, px: float | None, sz: float) -> None:
        """Validate minSz and precision constraints; raise PrecisionError on violation."""
        inst = self.get(instId)
        lot = _D(inst.lotSz)
        tick = _D(inst.tickSz)
        min_sz = _D(inst.minSz)
        sz_d = _D(sz)
        if sz_d < min_sz:
            raise PrecisionError(
                f"Size {sz_d} < minSz {min_sz} for {instId}",
                field="sz", expected=f">={min_sz}", actual=str(sz_d), instId=instId
            )

        if not _is_multiple(sz_d, lot):
            # 给出修复建议
            fixed = _floor_multiple(sz_d, lot)
            raise PrecisionError(
                f"Size {sz_d} not multiple of lotSz {lot} for {instId}",
                field="sz", expected=f"multiple of {lot}", actual=str(sz_d),
                suggestion=float(fixed), instId=instId
            )

        if px is not None:
            px_d = _D(px)
            if not _is_multiple(px_d, tick):
                fixed_px = _nearest_multiple(px_d, tick)
                raise PrecisionError(
                    f"Price {px_d} not multiple of tickSz {tick} for {instId}",
                    field="px", expected=f"multiple of {tick}", actual=str(px_d),
                    suggestion=float(fixed_px), instId=instId
                )
            