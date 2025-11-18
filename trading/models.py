# trading/models.py
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Callable, Awaitable, List
from enum import Enum
from trading.enums import PosSide, Side, TdMode, OrdType, TimeInForce, OrderStatus
from decimal import Decimal


@dataclass
class Instrument:
    instId: str
    tickSz: float # 价格步长 / 最小价格精度
    lotSz: float  # 数量步长
    minSz: float  # 最小下单数量
    ctVal: float  # 合约面值


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
    sz: str = ""
    px: Optional[str] = None
    tdMode: Literal["isolated","cross"] = "isolated"

    posSide: Optional[Literal["net","long","short"]] = "net"
    reduceOnly: Optional[bool] = None
    tag: Optional[str] = None
    clOrdId: Optional[str] = None
    expTime: Optional[int] = None
    
    leverage: Optional[int] = None


@dataclass
class OrderAck:
    instId: str
    clOrdId: str
    ordId: Optional[str]
    accepted: bool
    msg: Optional[str] = None

@dataclass
class OKXOrderFeed:
    ...

# Sell short 
"[ORDERS EVENT] {'arg': {'channel': 'orders', 'instType': 'SWAP', 'uid': '285627543871205376'}, 'data': [{'instType': 'SWAP', 'instId': 'DOGE-USDT-SWAP', 'tgtCcy': '', 'ccy': 'USDT', 'tradeQuoteCcy': '', 'ordId': '3040114693417193472', 'clOrdId': '', 'algoClOrdId': '', 'algoId': '', 'tag': '', 'px': '', 'sz': '0.11', 'notionalUsd': '18.355685039999997', 'ordType': 'market', 'side': 'sell', 'posSide': 'net', 'tdMode': 'isolated', 'accFillSz': '0', 'fillNotionalUsd': '', 'avgPx': '0', 'state': 'live', 'lever': '5', 'pnl': '0', 'feeCcy': 'USDT', 'fee': '0', 'rebateCcy': 'USDT', 'rebate': '0', 'category': 'normal', 'uTime': '1763104878188', 'cTime': '1763104878188', 'source': '', 'reduceOnly': 'false', 'cancelSource': '', 'quickMgnType': '', 'stpId': '', 'stpMode': 'cancel_maker', 'attachAlgoClOrdId': '', 'lastPx': '0.16695', 'isTpLimit': 'false', 'slTriggerPx': '', 'slTriggerPxType': '', 'tpOrdPx': '', 'tpTriggerPx': '', 'tpTriggerPxType': '', 'slOrdPx': '', 'fillPx': '', 'tradeId': '', 'fillSz': '0', 'fillTime': '', 'fillPnl': '0', 'fillFee': '0', 'fillFeeCcy': '', 'execType': '', 'fillPxVol': '', 'fillPxUsd': '', 'fillMarkVol': '', 'fillFwdPx': '', 'fillMarkPx': '', 'fillIdxPx': '', 'amendSource': '', 'reqId': '', 'amendResult': '', 'code': '0', 'msg': '', 'pxType': '', 'pxUsd': '', 'pxVol': '', 'linkedAlgoOrd': {'algoId': ''}, 'attachAlgoOrds': []}]}"
"[ORDERS EVENT] {'arg': {'channel': 'orders', 'instType': 'SWAP', 'uid': '285627543871205376'}, 'data': [{'instType': 'SWAP', 'instId': 'DOGE-USDT-SWAP', 'tgtCcy': '', 'ccy': 'USDT', 'tradeQuoteCcy': '', 'ordId': '3040114693417193472', 'clOrdId': '', 'algoClOrdId': '', 'algoId': '', 'tag': '', 'px': '', 'sz': '0.11', 'notionalUsd': '18.344690319999994', 'ordType': 'market', 'side': 'sell', 'posSide': 'net', 'tdMode': 'isolated', 'accFillSz': '0.11', 'fillNotionalUsd': '18.344690319999994', 'avgPx': '0.16685', 'state': 'filled', 'lever': '5', 'pnl': '0', 'feeCcy': 'USDT', 'fee': '-0.00917675', 'rebateCcy': 'USDT', 'rebate': '0', 'category': 'normal', 'uTime': '1763104878189', 'cTime': '1763104878188', 'source': '', 'reduceOnly': 'false', 'cancelSource': '', 'quickMgnType': '', 'stpId': '', 'stpMode': 'cancel_maker', 'attachAlgoClOrdId': '', 'lastPx': '0.16685', 'isTpLimit': 'false', 'slTriggerPx': '', 'slTriggerPxType': '', 'tpOrdPx': '', 'tpTriggerPx': '', 'tpTriggerPxType': '', 'slOrdPx': '', 'fillPx': '0.16685', 'tradeId': '1459457117', 'fillSz': '0.11', 'fillTime': '1763104878189', 'fillPnl': '0', 'fillFee': '-0.00917675', 'fillFeeCcy': 'USDT', 'execType': 'T', 'fillPxVol': '', 'fillPxUsd': '', 'fillMarkVol': '', 'fillFwdPx': '', 'fillMarkPx': '0.16682', 'fillIdxPx': '0.164906', 'amendSource': '', 'reqId': '', 'amendResult': '', 'code': '0', 'msg': '', 'pxType': '', 'pxUsd': '', 'pxVol': '', 'linkedAlgoOrd': {'algoId': ''}, 'attachAlgoOrds': []}]}"

# FLAT short position
"[ORDERS EVENT] {'arg': {'channel': 'orders', 'instType': 'SWAP', 'uid': '285627543871205376'}, 'data': [{'instType': 'SWAP', 'instId': 'DOGE-USDT-SWAP', 'tgtCcy': '', 'ccy': 'USDT', 'tradeQuoteCcy': '', 'ordId': '3040121372259110912', 'clOrdId': '', 'algoClOrdId': '', 'algoId': '', 'tag': '', 'px': '', 'sz': '0.11', 'notionalUsd': '18.37272855', 'ordType': 'market', 'side': 'buy', 'posSide': 'net', 'tdMode': 'isolated', 'accFillSz': '0', 'fillNotionalUsd': '', 'avgPx': '0', 'state': 'live', 'lever': '5', 'pnl': '0', 'feeCcy': 'USDT', 'fee': '0', 'rebateCcy': 'USDT', 'rebate': '0', 'category': 'normal', 'uTime': '1763105077233', 'cTime': '1763105077233', 'source': '', 'reduceOnly': 'true', 'cancelSource': '', 'quickMgnType': '', 'stpId': '', 'stpMode': 'cancel_maker', 'attachAlgoClOrdId': '', 'lastPx': '0.1671', 'isTpLimit': 'false', 'tpTriggerPx': '0', 'tpTriggerPxType': '', 'tpOrdPx': '0', 'slTriggerPx': '0', 'slOrdPx': '0', 'slTriggerPxType': '', 'fillPx': '', 'tradeId': '', 'fillSz': '0', 'fillTime': '', 'fillPnl': '0', 'fillFee': '0', 'fillFeeCcy': '', 'execType': '', 'fillPxVol': '', 'fillPxUsd': '', 'fillMarkVol': '', 'fillFwdPx': '', 'fillMarkPx': '', 'fillIdxPx': '', 'amendSource': '', 'reqId': '', 'amendResult': '', 'code': '0', 'msg': '', 'pxType': '', 'pxUsd': '', 'pxVol': '', 'linkedAlgoOrd': {'algoId': ''}, 'attachAlgoOrds': []}]}"
"[ORDERS EVENT] {'arg': {'channel': 'orders', 'instType': 'SWAP', 'uid': '285627543871205376'}, 'data': [{'instType': 'SWAP', 'instId': 'DOGE-USDT-SWAP', 'tgtCcy': '', 'ccy': 'USDT', 'tradeQuoteCcy': '', 'ordId': '3040121372259110912', 'clOrdId': '', 'algoClOrdId': '', 'algoId': '', 'tag': '', 'px': '', 'sz': '0.11', 'notionalUsd': '18.3837236', 'ordType': 'market', 'side': 'buy', 'posSide': 'net', 'tdMode': 'isolated', 'accFillSz': '0.11', 'fillNotionalUsd': '18.3837236', 'avgPx': '0.1672', 'state': 'filled', 'lever': '5', 'pnl': '-0.0385', 'feeCcy': 'USDT', 'fee': '-0.009196', 'rebateCcy': 'USDT', 'rebate': '0', 'category': 'normal', 'uTime': '1763105077234', 'cTime': '1763105077233', 'source': '', 'reduceOnly': 'true', 'cancelSource': '', 'quickMgnType': '', 'stpId': '', 'stpMode': 'cancel_maker', 'attachAlgoClOrdId': '', 'lastPx': '0.1672', 'isTpLimit': 'false', 'tpTriggerPx': '', 'tpTriggerPxType': '', 'tpOrdPx': '', 'slTriggerPx': '', 'slOrdPx': '', 'slTriggerPxType': '', 'fillPx': '0.1672', 'tradeId': '1459478292', 'fillSz': '0.11', 'fillTime': '1763105077234', 'fillPnl': '-0.0385', 'fillFee': '-0.009196', 'fillFeeCcy': 'USDT', 'execType': 'T', 'fillPxVol': '', 'fillPxUsd': '', 'fillMarkVol': '', 'fillFwdPx': '', 'fillMarkPx': '0.16707', 'fillIdxPx': '0.165192', 'amendSource': '', 'reqId': '', 'amendResult': '', 'code': '0', 'msg': '', 'pxType': '', 'pxUsd': '', 'pxVol': '', 'linkedAlgoOrd': {'algoId': ''}, 'attachAlgoOrds': []}]}"

# Buy long
"[ORDERS EVENT] {'arg': {'channel': 'orders', 'instType': 'SWAP', 'uid': '285627543871205376'}, 'data': [{'instType': 'SWAP', 'instId': 'DOGE-USDT-SWAP', 'tgtCcy': '', 'ccy': 'USDT', 'tradeQuoteCcy': '', 'ordId': '3040123512394338304', 'clOrdId': '', 'algoClOrdId': '', 'algoId': '', 'tag': '', 'px': '', 'sz': '0.11', 'notionalUsd': '18.37217712', 'ordType': 'market', 'side': 'buy', 'posSide': 'net', 'tdMode': 'isolated', 'accFillSz': '0', 'fillNotionalUsd': '', 'avgPx': '0', 'state': 'live', 'lever': '5', 'pnl': '0', 'feeCcy': 'USDT', 'fee': '0', 'rebateCcy': 'USDT', 'rebate': '0', 'category': 'normal', 'uTime': '1763105141014', 'cTime': '1763105141014', 'source': '', 'reduceOnly': 'false', 'cancelSource': '', 'quickMgnType': '', 'stpId': '', 'stpMode': 'cancel_maker', 'attachAlgoClOrdId': '', 'lastPx': '0.1671', 'isTpLimit': 'false', 'slTriggerPx': '', 'slTriggerPxType': '', 'tpOrdPx': '', 'tpTriggerPx': '', 'tpTriggerPxType': '', 'slOrdPx': '', 'fillPx': '', 'tradeId': '', 'fillSz': '0', 'fillTime': '', 'fillPnl': '0', 'fillFee': '0', 'fillFeeCcy': '', 'execType': '', 'fillPxVol': '', 'fillPxUsd': '', 'fillMarkVol': '', 'fillFwdPx': '', 'fillMarkPx': '', 'fillIdxPx': '', 'amendSource': '', 'reqId': '', 'amendResult': '', 'code': '0', 'msg': '', 'pxType': '', 'pxUsd': '', 'pxVol': '', 'linkedAlgoOrd': {'algoId': ''}, 'attachAlgoOrds': []}]}"
"[ORDERS EVENT] {'arg': {'channel': 'orders', 'instType': 'SWAP', 'uid': '285627543871205376'}, 'data': [{'instType': 'SWAP', 'instId': 'DOGE-USDT-SWAP', 'tgtCcy': '', 'ccy': 'USDT', 'tradeQuoteCcy': '', 'ordId': '3040123512394338304', 'clOrdId': '', 'algoClOrdId': '', 'algoId': '', 'tag': '', 'px': '', 'sz': '0.11', 'notionalUsd': '18.37767448', 'ordType': 'market', 'side': 'buy', 'posSide': 'net', 'tdMode': 'isolated', 'accFillSz': '0.11', 'fillNotionalUsd': '18.37767448', 'avgPx': '0.16715', 'state': 'filled', 'lever': '5', 'pnl': '0', 'feeCcy': 'USDT', 'fee': '-0.00919325', 'rebateCcy': 'USDT', 'rebate': '0', 'category': 'normal', 'uTime': '1763105141015', 'cTime': '1763105141014', 'source': '', 'reduceOnly': 'false', 'cancelSource': '', 'quickMgnType': '', 'stpId': '', 'stpMode': 'cancel_maker', 'attachAlgoClOrdId': '', 'lastPx': '0.16715', 'isTpLimit': 'false', 'slTriggerPx': '', 'slTriggerPxType': '', 'tpOrdPx': '', 'tpTriggerPx': '', 'tpTriggerPxType': '', 'slOrdPx': '', 'fillPx': '0.16715', 'tradeId': '1459488728', 'fillSz': '0.11', 'fillTime': '1763105141015', 'fillPnl': '0', 'fillFee': '-0.00919325', 'fillFeeCcy': 'USDT', 'execType': 'T', 'fillPxVol': '', 'fillPxUsd': '', 'fillMarkVol': '', 'fillFwdPx': '', 'fillMarkPx': '0.16717', 'fillIdxPx': '0.165169', 'amendSource': '', 'reqId': '', 'amendResult': '', 'code': '0', 'msg': '', 'pxType': '', 'pxUsd': '', 'pxVol': '', 'linkedAlgoOrd': {'algoId': ''}, 'attachAlgoOrds': []}]}"

# Flat long position
"[ORDERS EVENT] {'arg': {'channel': 'orders', 'instType': 'SWAP', 'uid': '285627543871205376'}, 'data': [{'instType': 'SWAP', 'instId': 'DOGE-USDT-SWAP', 'tgtCcy': '', 'ccy': 'USDT', 'tradeQuoteCcy': '', 'ordId': '3040124862725345280', 'clOrdId': '', 'algoClOrdId': '', 'algoId': '', 'tag': '', 'px': '', 'sz': '0.11', 'notionalUsd': '18.46031957', 'ordType': 'market', 'side': 'sell', 'posSide': 'net', 'tdMode': 'isolated', 'accFillSz': '0', 'fillNotionalUsd': '', 'avgPx': '0', 'state': 'live', 'lever': '5', 'pnl': '0', 'feeCcy': 'USDT', 'fee': '0', 'rebateCcy': 'USDT', 'rebate': '0', 'category': 'normal', 'uTime': '1763105181257', 'cTime': '1763105181257', 'source': '', 'reduceOnly': 'true', 'cancelSource': '', 'quickMgnType': '', 'stpId': '', 'stpMode': 'cancel_maker', 'attachAlgoClOrdId': '', 'lastPx': '0.1679', 'isTpLimit': 'false', 'tpTriggerPx': '0', 'tpTriggerPxType': '', 'tpOrdPx': '0', 'slTriggerPx': '0', 'slOrdPx': '0', 'slTriggerPxType': '', 'fillPx': '', 'tradeId': '', 'fillSz': '0', 'fillTime': '', 'fillPnl': '0', 'fillFee': '0', 'fillFeeCcy': '', 'execType': '', 'fillPxVol': '', 'fillPxUsd': '', 'fillMarkVol': '', 'fillFwdPx': '', 'fillMarkPx': '', 'fillIdxPx': '', 'amendSource': '', 'reqId': '', 'amendResult': '', 'code': '0', 'msg': '', 'pxType': '', 'pxUsd': '', 'pxVol': '', 'linkedAlgoOrd': {'algoId': ''}, 'attachAlgoOrds': []}]}"
"[ORDERS EVENT] {'arg': {'channel': 'orders', 'instType': 'SWAP', 'uid': '285627543871205376'}, 'data': [{'instType': 'SWAP', 'instId': 'DOGE-USDT-SWAP', 'tgtCcy': '', 'ccy': 'USDT', 'tradeQuoteCcy': '', 'ordId': '3040124862725345280', 'clOrdId': '', 'algoClOrdId': '', 'algoId': '', 'tag': '', 'px': '', 'sz': '0.11', 'notionalUsd': '18.46031957', 'ordType': 'market', 'side': 'sell', 'posSide': 'net', 'tdMode': 'isolated', 'accFillSz': '0.11', 'fillNotionalUsd': '18.46031957', 'avgPx': '0.1679', 'state': 'filled', 'lever': '5', 'pnl': '0.0825', 'feeCcy': 'USDT', 'fee': '-0.0092345', 'rebateCcy': 'USDT', 'rebate': '0', 'category': 'normal', 'uTime': '1763105181258', 'cTime': '1763105181257', 'source': '', 'reduceOnly': 'true', 'cancelSource': '', 'quickMgnType': '', 'stpId': '', 'stpMode': 'cancel_maker', 'attachAlgoClOrdId': '', 'lastPx': '0.1679', 'isTpLimit': 'false', 'tpTriggerPx': '', 'tpTriggerPxType': '', 'tpOrdPx': '', 'slTriggerPx': '', 'slOrdPx': '', 'slTriggerPxType': '', 'fillPx': '0.1679', 'tradeId': '1459490304', 'fillSz': '0.11', 'fillTime': '1763105181258', 'fillPnl': '0.0825', 'fillFee': '-0.0092345', 'fillFeeCcy': 'USDT', 'execType': 'T', 'fillPxVol': '', 'fillPxUsd': '', 'fillMarkVol': '', 'fillFwdPx': '', 'fillMarkPx': '0.16783', 'fillIdxPx': '0.16518', 'amendSource': '', 'reqId': '', 'amendResult': '', 'code': '0', 'msg': '', 'pxType': '', 'pxUsd': '', 'pxVol': '', 'linkedAlgoOrd': {'algoId': ''}, 'attachAlgoOrds': []}]}"

# Place and cancel
"[ORDERS EVENT] {'arg': {'channel': 'orders', 'instType': 'SWAP', 'uid': '285627543871205376'}, 'data': [{'instType': 'SWAP', 'instId': 'BTC-USDT-SWAP', 'tgtCcy': '', 'ccy': 'USDT', 'tradeQuoteCcy': '', 'ordId': '3051152589448273920', 'clOrdId': '176343383288728824f962c6c48cd806', 'algoClOrdId': '', 'algoId': '', 'tag': '', 'px': '85000.1', 'sz': '0.01', 'notionalUsd': '8.490319988600001', 'ordType': 'limit', 'side': 'buy', 'posSide': 'net', 'tdMode': 'cross', 'accFillSz': '0', 'fillNotionalUsd': '', 'avgPx': '0', 'state': 'live', 'lever': '3', 'pnl': '0', 'feeCcy': 'USDT', 'fee': '0', 'rebateCcy': 'USDT', 'rebate': '0', 'category': 'normal', 'uTime': '1763433833124', 'cTime': '1763433833124', 'source': '', 'reduceOnly': 'false', 'cancelSource': '', 'quickMgnType': '', 'stpId': '', 'stpMode': 'cancel_maker', 'attachAlgoClOrdId': '', 'lastPx': '91480', 'isTpLimit': 'false', 'slTriggerPx': '', 'slTriggerPxType': '', 'tpOrdPx': '', 'tpTriggerPx': '', 'tpTriggerPxType': '', 'slOrdPx': '', 'fillPx': '', 'tradeId': '', 'fillSz': '0', 'fillTime': '', 'fillPnl': '0', 'fillFee': '0', 'fillFeeCcy': '', 'execType': '', 'fillPxVol': '', 'fillPxUsd': '', 'fillMarkVol': '', 'fillFwdPx': '', 'fillMarkPx': '', 'fillIdxPx': '', 'amendSource': '', 'reqId': '', 'amendResult': '', 'code': '0', 'msg': '', 'pxType': '', 'pxUsd': '', 'pxVol': '', 'linkedAlgoOrd': {'algoId': ''}, 'attachAlgoOrds': []}]}"
"[ORDERS EVENT] {'arg': {'channel': 'orders', 'instType': 'SWAP', 'uid': '285627543871205376'}, 'data': [{'instType': 'SWAP', 'instId': 'BTC-USDT-SWAP', 'tgtCcy': '', 'ccy': 'USDT', 'tradeQuoteCcy': '', 'ordId': '3051152589448273920', 'clOrdId': '176343383288728824f962c6c48cd806', 'algoClOrdId': '', 'algoId': '', 'tag': '', 'px': '85000.1', 'sz': '0.01', 'notionalUsd': '8.490319988600001', 'ordType': 'limit', 'side': 'buy', 'posSide': 'net', 'tdMode': 'cross', 'accFillSz': '0', 'fillNotionalUsd': '', 'avgPx': '0', 'state': 'canceled', 'lever': '3', 'pnl': '0', 'feeCcy': 'USDT', 'fee': '0', 'rebateCcy': 'USDT', 'rebate': '0', 'category': 'normal', 'uTime': '1763433833291', 'cTime': '1763433833124', 'source': '', 'reduceOnly': 'false', 'cancelSource': '1', 'quickMgnType': '', 'stpId': '', 'stpMode': 'cancel_maker', 'attachAlgoClOrdId': '', 'lastPx': '91480', 'isTpLimit': 'false', 'slTriggerPx': '', 'slTriggerPxType': '', 'tpOrdPx': '', 'tpTriggerPx': '', 'tpTriggerPxType': '', 'slOrdPx': '', 'fillPx': '', 'tradeId': '', 'fillSz': '0', 'fillTime': '', 'fillPnl': '0', 'fillFee': '0', 'fillFeeCcy': '', 'execType': '', 'fillPxVol': '', 'fillPxUsd': '', 'fillMarkVol': '', 'fillFwdPx': '', 'fillMarkPx': '', 'fillIdxPx': '', 'amendSource': '', 'reqId': '', 'amendResult': '', 'code': '0', 'msg': '', 'pxType': '', 'pxUsd': '', 'pxVol': '', 'linkedAlgoOrd': {'algoId': ''}, 'attachAlgoOrds': []}]}"