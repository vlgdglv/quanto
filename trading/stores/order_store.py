# trading/stores/order_store.py
from typing import Optional, Dict
from trading.models import Order

class OrderStore:
    """
    In-memory order mirror keyed by clOrdId and ordId.
    """

    def __init__(self) -> None:
        self._by_cl: Dict[str, Order] = {}
        self._by_id: Dict[str, Order] = {}

    def upsert(self, order: Order) -> None:
        """Insert or update an order mirror."""
        self._by_cl[order.clOrdId] = order
        if order.ordId:
            self._by_id[order.ordId] = order

    def get_by_cl(self, clOrdId: str) -> Optional[Order]:
        return self._by_cl.get(clOrdId)

    def get_by_id(self, ordId: str) -> Optional[Order]:
        return self._by_id.get(ordId)
