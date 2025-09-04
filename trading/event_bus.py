# trading/event_bus.py
import asyncio
from typing import Any, Callable, Dict

class EventBus:
    """
    Lightweight async pub/sub for order/position/balance updates.
    """

    def __init__(self) -> None:
        self._subs: Dict[str, list[Callable[[Any], None]]] = {}

    def subscribe(self, topic: str, handler: Callable[[Any], None]) -> None:
        """Register a synchronous callback; wrap async handlers externally."""
        self._subs.setdefault(topic, []).append(handler)

    def publish(self, topic: str, payload: Any) -> None:
        """Publish an event to subscribers (fire-and-forget)."""
        for h in self._subs.get(topic, []):
            try:
                h(payload)
            except Exception:
                # TODO: route to structured logger
                pass

# Common topics
TOPIC_ORDER = "order.update"
TOPIC_POSITION = "position.update"
TOPIC_BALANCE = "balance.update"
