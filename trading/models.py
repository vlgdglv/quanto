# trading/models.py
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Callable, Awaitable, List
from enum import Enum
from trading.enums import PosSide, Side, TdMode, OrdType, TimeInForce, OrderStatus
from decimal import Decimal

@dataclass
class Instrument:
    instId: str
    tickSz: Decimal
    lotSz: Decimal
    minSz: Decimal
    ctVal: Decimal

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
    ccy: str                  # 币种
    equity: float             # details.eq: 总权益/总余额
    avail: float              # 优先 details.availBal；若为空用 details.availEq
    frozen: float             # details.frozenBal（若无可用 0）
    ts: int        


@dataclass
class ModeTarget:
    acctLv: Optional[str] = None          # 账户级别或保证金模式（按需）
    posMode: Optional[str] = None         # "net_mode" / "long_short_mode"
    mgnMode: Optional[str] = None         # "cross" / "isolated"
    leverage: Optional[float] = None      # 目标杠杆（合约层面可能需 per instId 设置）


class OrdState(str, Enum):
    NEW="new"
    LIVE="live"
    PARTIALLY_FILLED="partially_filled"
    FILLED="filled"
    CANCELED="canceled"
    FAILED="failed"


@dataclass
class OrderCmd:
    instId: str
    side: Literal["buy","sell"]
    ordType: Literal["limit","market","post_only","ioc","fok","optimal_limit_ioc"]
    tdMode: Literal["isolated","cross"] = "cross"
    posSide: Optional[Literal["net","long","short"]] = "net"
    sz: str = ""
    px: Optional[str] = None
    reduceOnly: Optional[bool] = None
    tag: Optional[str] = None
    clOrdId: Optional[str] = None
    expTime: Optional[int] = None


@dataclass
class OrderAck:
    instId: str
    clOrdId: str
    ordId: Optional[str]
    accepted: bool
    msg: Optional[str] = None
