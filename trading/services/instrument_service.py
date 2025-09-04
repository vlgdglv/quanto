# trading/services/instrument_service.py
from typing import Dict
from trading.models import Instrument
from trading.errors import PrecisionError

class InstrumentService:
    """
    Pulls & caches instrument specs (tickSz/lotSz/minSz/ctVal) for SWAP,
    and provides normalization/validation helpers.
    """

    def __init__(self, http_client, endpoints) -> None:
        self._http = http_client
        self._ep = endpoints
        self._cache: Dict[str, Instrument] = {}

    async def refresh(self) -> None:
        """Fetch /public/instruments?instType=SWAP and populate cache."""
        # TODO: request, parse, fill self._cache
        ...

    def get(self, instId: str) -> Instrument:
        """Return instrument specs or raise if unknown."""
        inst = self._cache.get(instId)
        if not inst:
            raise KeyError(f"Unknown instrument: {instId}")
        return inst

    def round_price(self, instId: str, px: float) -> float:
        """Round price to tickSz."""
        # TODO: implement rounding
        ...

    def normalize_size(self, instId: str, sz: float) -> float:
        """Round size to lotSz multiples."""
        # TODO: implement size normalization
        ...

    def validate(self, instId: str, px: float | None, sz: float) -> None:
        """Validate minSz and precision constraints; raise PrecisionError on violation."""
        # TODO: implement precision/min checks
        ...
