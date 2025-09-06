# trading/models.py
from dataclasses import dataclass
from typing import Optional, Dict
from trading.enums import PosSide, Side, TdMode, OrdType, TimeInForce, OrderStatus
from decimal import Decimal

@dataclass
class Instrument:
    instId: str
    tickSz: Decimal
    lotSz: Decimal
    minSz: Decimal
    ctVal: Decimal
    # ts: int

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
    # --- 基本识别 ---
    instType: str           # "SWAP"
    instId: str             # e.g. "BTC-USDT-SWAP"
    posId: str
    posSide: PosSide        # "long" | "short" | "net"
    mgnMode: str            # "cross" | "isolated"

    # --- 数量与价格 ---
    pos: float              # 仓位数量
    availPos: float         # 可平数量（OKX返回为空串时视为 0）
    avgPx: Optional[float]  # 开仓均价
    markPx: Optional[float] # 标记价格
    liqPx: Optional[float]  # 预估强平价
    lever: Optional[float]  # 杠杆倍数（期权等可能为空）

    # --- 盈亏与名义价值 ---
    upl: Optional[float]        # 以标记价计算的未实现盈亏
    uplRatio: Optional[float]   # 未实现收益率
    notionalUsd: Optional[float]

    # --- 保证金相关 ---
    imr: Optional[float]        # 初始保证金（全仓）
    mmr: Optional[float]        # 维持保证金
    margin: Optional[float]     # 逐仓保证金余额
    mgnRatio: Optional[float]   # 逐仓保证金率

    # --- 其他展示/风控 ---
    adl: Optional[int]      # 1-5，数值越小adl强度越弱
    cTime: Optional[int]    # 创建时间(ms)
    uTime: Optional[int]    # 最近更新时间(ms)

@dataclass
class Balance:
    ccy: str
    equity: float
    availEq: float
    ts: int


@dataclass
class ModeTarget:
    acctLv: Optional[str] = None          # 账户级别或保证金模式（按需）
    posMode: Optional[str] = None         # "net_mode" / "long_short_mode"
    mgnMode: Optional[str] = None         # "cross" / "isolated"
    leverage: Optional[float] = None      # 目标杠杆（合约层面可能需 per instId 设置）
