# trading/services/reconcile_service.py
from ..stores.order_store import OrderStore
from ..errors import ReconcileTimeout
from ..models import Order

class ReconcileService:
    """
    REST+WS reconciliation keyed by clOrdId/ordId to ensure final consistency.
    """

    def __init__(self, exec_service, order_store: OrderStore, logger) -> None:
        self._exec = exec_service
        self._orders = order_store
        self._log = logger

    async def reconcile_by_clOrdId(self, clOrdId: str, timeout_s: float = 3.0) -> Order:
        """
        Wait for ACK/final state or timeout, using:
        - polling GET /trade/order
        - checking OrderStore updates (possibly via EventBus externally)
        """
        ...

    async def periodic_reconcile(self) -> None:
        """Background task to reconcile dangling orders."""
        ...
