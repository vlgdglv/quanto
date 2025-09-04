# trading/services/risk_service.py
from ..models import OrderRequest, Position
from ..errors import PrecisionError, MarginError, ReduceOnlyError, SlippageError

class RiskService:
    """
    Pre-trade checks: precision, min/max size, margin/availability, reduce-only, slippage.
    """

    def __init__(self, instrument_svc, account_svc, logger) -> None:
        self._inst = instrument_svc
        self._acct = account_svc
        self._log = logger

    def check_precision(self, req: OrderRequest) -> None:
        """Validate price/size against tickSz/lotSz/minSz."""
        ...

    async def check_margins(self, req: OrderRequest) -> None:
        """Ensure sufficient margin or max size; raise MarginError if violates."""
        ...

    def check_slippage(self, best_bid: float, best_ask: float, req: OrderRequest, max_bps: float) -> None:
        """Ensure requested price within allowable slippage from book; raise SlippageError."""
        ...

    def enforce_reduce_only(self, req: OrderRequest, current_pos: Position | None) -> None:
        """Ensure reduceOnly orders do not increase exposure; raise ReduceOnlyError."""
        ...
