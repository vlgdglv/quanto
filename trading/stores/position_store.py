# trading/stores/position_store.py
from typing import Dict, List
from trading.models import Position

class PositionStore:
    """
    In-memory position mirror keyed by (instId, posSide).
    """

    def __init__(self) -> None:
        self._data: Dict[tuple[str, str], Position] = {}

    def upsert(self, pos: Position) -> None:
        self._data[(pos.instId, pos.posSide)] = pos

    def list_all(self) -> List[Position]:
        return list(self._data.values())
