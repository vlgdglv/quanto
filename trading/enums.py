# trading/enums.py
from enum import Enum

class Env(Enum):
    PAPER = "paper"
    LIVE = "live"

class Side(Enum):
    BUY = "buy"
    SELL = "sell"

class PosMode(Enum):
    NET = "net"
    LONG_SHORT = "long_short"

class TdMode(Enum):
    CROSS = "cross"
    ISOLATED = "isolated"

class OrdType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    POST_ONLY = "post_only"
    FOK = "fok"
    IOC = "ioc"

class TimeInForce(Enum):
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"

class OrderStatus(Enum):
    NEW = "new"
    SENT = "sent"
    ACK = "ack"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
