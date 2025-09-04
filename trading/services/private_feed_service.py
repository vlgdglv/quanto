# trading/services/private_feed_service.py
from ..event_bus import EventBus, TOPIC_ORDER, TOPIC_POSITION, TOPIC_BALANCE
from ..stores.order_store import OrderStore
from ..stores.position_store import PositionStore
from ..stores.balance_store import BalanceStore

class PrivateFeedService:
    """
    Private WebSocket client: login, subscribe to orders/positions/account,
    normalize payloads and fan-out to stores and event bus.
    """

    def __init__(self, ws_client, endpoints, order_store: OrderStore,
                 position_store: PositionStore, balance_store: BalanceStore,
                 event_bus: EventBus, logger) -> None:
        self._ws = ws_client
        self._ep = endpoints
        self._orders = order_store
        self._positions = position_store
        self._balances = balance_store
        self._bus = event_bus
        self._log = logger

    async def connect_and_login(self) -> None:
        """Open WS connection and send login frame with API key/secret/passphrase."""
        ...

    async def subscribe_orders(self) -> None:
        """Subscribe to 'orders' channel; route updates to OrderStore & EventBus."""
        ...

    async def subscribe_account_positions(self) -> None:
        """Subscribe to 'account' and 'positions' channels."""
        ...

    # Internal handlers (examples)
    async def _on_order_update(self, payload: dict) -> None:
        """Parse and upsert order; publish event."""
        ...

    async def _on_position_update(self, payload: dict) -> None:
        """Parse and upsert position; publish event."""
        ...

    async def _on_balance_update(self, payload: dict) -> None:
        """Parse and upsert balance; publish event."""
        ...
