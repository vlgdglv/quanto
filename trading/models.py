# trading/models.py
from dataclasses import dataclass
from typing import Optional, Dict
from trading.enums import Side, TdMode, OrdType, TimeInForce, OrderStatus

@dataclass
class Instrument:
    instId: str
    tickSz: float
    lotSz: float
    minSz: float
    ctVal: float
    ts: int

@dataclass
class OrderRequest:
    instId: str
    side: Side
    tdMode: TdMode
    posSide: Optional[str]              # "net" | "long" | "short"
    ordType: OrdType
    sz: float
    px: Optional[float] = None
    tif: Optional[TimeInForce] = None
    reduceOnly: bool = False
    clOrdId: Optional[str] = None
    expTime: Optional[int] = None
    attach_tp_px: Optional[float] = None
    attach_sl_px: Optional[float] = None
    tags: Optional[Dict[str, str]] = None

@dataclass
class Order:
    clOrdId: str
    ordId: Optional[str]
    req: OrderRequest
    status: OrderStatus
    filledSz: float = 0.0
    avgPx: Optional[float] = None
    createTs: int = 0
    updateTs: int = 0
    raw: Optional[dict] = None

@dataclass
class Fill:
    ordId: str
    instId: str
    px: float
    sz: float
    side: Side
    fee: float
    ts: int
    raw: Optional[dict] = None

@dataclass
class Position:
    instId: str
    posSide: str
    sz: float
    avgPx: float
    upl: float
    ts: int

@dataclass
class Balance:
    ccy: str
    equity: float
    availEq: float
    ts: int
