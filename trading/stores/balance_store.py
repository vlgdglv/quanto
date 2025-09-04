# trading/stores/balance_store.py
from typing import Dict, Optional
from ..models import Balance

class BalanceStore:
    """
    In-memory balance mirror keyed by currency code.
    """

    def __init__(self) -> None:
        self._data: Dict[str, Balance] = {}

    def upsert(self, bal: Balance) -> None:
        self._data[bal.ccy] = bal

    def get(self, ccy: str) -> Optional[Balance]:
        return self._data.get(ccy)
