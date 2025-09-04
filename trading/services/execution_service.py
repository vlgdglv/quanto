# trading/services/execution_service.py
from typing import List, Optional
from ..models import OrderRequest, Order, Fill
from ..errors import OkxApiError

class ExecutionService:
    """
    REST-based order placement/cancel/amend and queries.
    """

    def __init__(self, http_client, endpoints, order_store, logger) -> None:
        self._http = http_client
        self._ep = endpoints
        self._orders = order_store
        self._log = logger

    async def place_order(self, req: OrderRequest) -> Order:
        """POST /trade/order → create local Order and store."""
        ...

    async def cancel_order(self, instId: str, ordId: Optional[str] = None, clOrdId: Optional[str] = None) -> Order:
        """POST /trade/cancel-order → update local Order."""
        ...

    async def amend_order(self, instId: str, ordId: Optional[str], new_px: Optional[float], new_sz: Optional[float]) -> Order:
        """POST /trade/amend-order → update local Order."""
        ...

    async def get_order(self, instId: str, ordId: Optional[str] = None, clOrdId: Optional[str] = None) -> Order:
        """GET /trade/order → refresh a single Order."""
        ...

    async def list_open_orders(self, instId: Optional[str] = None) -> List[Order]:
        """GET /trade/orders-pending → return and update mirrors."""
        ...

    async def list_fills(self, instId: str, after: Optional[int] = None, limit: int = 100) -> List[Fill]:
        """GET /trade/fills-history → parse into Fill list."""
        ...
