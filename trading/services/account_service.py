# trading/services/account_service.py
from typing import List, Optional
from ..models import Position, Balance
from ..enums import TdMode

class AccountService:
    """
    Account/position/leverage/mode configuration and queries.
    """

    def __init__(self, http_client, endpoints) -> None:
        self._http = http_client
        self._ep = endpoints

    async def get_config(self) -> dict:
        """Return account config (position mode, Greeks, etc.)."""
        ...

    async def set_position_mode(self, net: bool) -> None:
        """Set position mode to net or long/short; requires no open pos/orders."""
        ...

    async def set_leverage(self, instId: str, lever: int, mgnMode: TdMode) -> None:
        """Set leverage for an instrument and margin mode."""
        ...

    async def get_max_avail_size(self, instId: str, mgnMode: TdMode, ccy: Optional[str] = None) -> float:
        """Return maximum available size for order placement."""
        ...

    async def get_positions(self, instType: str = "SWAP") -> List[Position]:
        """Return current positions for instType."""
        ...

    async def get_balance(self, ccy: str = "USDT") -> Balance:
        """Return balance object for given currency."""
        ...
