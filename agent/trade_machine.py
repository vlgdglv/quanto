# tradeflow/engine.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, List
import time, math
from typing import Optional, List, Tuple, Literal, Dict, Any
from enum import Enum, auto

from agent.schemas import TradePlan, Side
from trading.models import OrderCmd, Position, Balance, MarketTicker, Instrument
from trading.services.account_service import AccountService
from trading.services.trading_service import TradingService, OrdersFeed
from trading.services.instrument_service import InstrumentService


from utils.logger import logger

@dataclass
class NewOrderRequest:
    inst: str
    side: str
    size: float
    price: float 

@dataclass
class CloseRequest:
    inst: str
    size: float
    reason: str


class PositionState(Enum):
    """High-level lifecycle state per instrument."""
    # MVP
    NO_POSITION = auto()   # 空仓
    OPEN = auto()          # 有持仓
    # Futures
    OPENING = auto()       # 已生成计划，还未下单或等待成交
    CLOSING = auto()       # 正在平仓（挂了平仓单，等待成交）
    CLOSED = auto()        # 最近一笔仓位刚刚结束（可用于复盘）


@dataclass
class PositionPlan:
    inst: str
    side: Optional[Side] = None
    size: float = None

    created_ts: float = None    

    stop_price: Optional[float] = None    # 硬止损
    tp_price: Optional[float]   = None    # 第一止盈
    
    horizon_sec: Optional[int]  = None    # 期望最大持仓时间（超过则 time-stop）
    min_hold_sec: Optional[int] = None    # 最短持仓时间（避免刚进就被噪声震出）

    avg_entry_price: Optional[float] = None

    opened_ts: Optional[float] = None
    closed_ts: Optional[float] = None

    last_price: Optional[float] = None
    floating_pnl: Optional[float] = 0.0

    realized_pnl: Optional[float] = None
    meta: Optional[Dict[str, Any]] = field(default_factory=dict)

    def holding_duration_sec(self, now: Optional[float] = None) -> Optional[float]:
        if self.opened_ts is None:
            return None
        if now is None:
            now = time.time()
        return now - self.opened_ts

class TradeMachine:
    """
    MVP级别的单品种交易自动机：
    - 只跟踪一个 inst 的单一净仓位（net position）
    - 不负责撮合，只产生命令 NewOrderRequest / CloseRequest
    """
    def __init__(
        self,
        inst: str,
        account_service: AccountService,
        trading_service: TradingService,
        instrument_service: InstrumentService,
        orders_feed: Optional[OrdersFeed] = None, 
        trading_ccy: str = "USDT",
        leverage: int = 10, 
        *,
        on_new_order: Callable[[NewOrderRequest], None] = None,
        on_close: Callable[[CloseRequest], None] = None,
    ) -> None:
        self.inst = inst
        # User must ensure initial state is NO_POSITION
        self.state: PositionState = PositionState.NO_POSITION
        self.postion_state = PositionPlan(inst=inst)

        self._on_new_order = on_new_order
        self._on_close = on_close
        # self._log = logger or (lambda msg: None)

        self.leverage = leverage
        self.ccy = trading_ccy
        self.account_service = account_service
        self.trading_service = trading_service
        self.instrument_service = instrument_service
        
        # Multiple pending order
        # self._pending_open_order_id: Optional[Dict[str, PendingOrder]] = None
        # self._pending_close_order_id: Optional[Dict[str, str, PendingOrder]] = None
        
        self._pending_open_order_id: Optional[str] = None
        self._pending_close_order_id: Optional[str] = None
        
    async def _transition_to(self, target_state: PositionState, payload: Optional[Dict[str, Any]] = None):
        current_state = self.state
        
        if not self._is_valid_transition(current_state, target_state):
            logger.error(f"Attempted invalid transition from {current_state} to {target_state}")
        
        # logger.info(f"[STATE CHANGE] {current_state} -> {target_state}, payload={payload}")
        
        # Maybe some hooks here
        self.state = target_state
        if payload:
            self._update_position_data(payload)
            
        # Post hook for state change
        await self._on_state_change(current_state, target_state, payload)
        
    def _is_valid_transition(self, current_state: PositionState, target_state: PositionState):
        # For simplicity, assume all transitions are valid
        return True
    
    def _update_position_data(self, payload: Dict[str, Any]):
        pass
    
    async def _on_state_change(self, current_state: PositionState, target_state: PositionState, payload: Optional[Dict[str, Any]] = None):
        pass
        
    def get_state(self):
        return self.state
    
    async def step(self,
                   decisions,
                   position: List[Position],
                   ):
        if self.state == PositionState.NO_POSITION:
            await self._handle_no_position(decisions.action)
        elif self.state == PositionState.OPEN:
            await self._handle_open_position(decisions.action)
        else:
            raise NotImplementedError
        
    async def _handle_no_position(self, 
                                  action: str,  # ["BUY", "SELL", "SKIP"]
                                 ):
        
        if action == "STALK":
            return
        
        side = "buy" if action.upper() == "OPEN_LONG" else "sell"
        
        try:
            balance_list: List[Balance] = await self.account_service.get_balance(ccy=self.ccy)
            balance = None
            for _b in balance_list:
                if _b.ccy == self.ccy:
                    balance = _b
                    break
            
            if not balance:
                logger.warning(f"Cannot find balance for {self.ccy}, skipping order.")
                return
            
            available_balance_str = balance.avail

            logger.debug(f"Available balance for {self.ccy}: {available_balance_str}")
            
            marker_price: MarketTicker = await self.instrument_service.get_inst_price(self.inst)
            instrument: Instrument = await self.instrument_service.get_or_refresh(self.inst)
            target_size, target_price = self._calculate_order_size(available_balance_str, marker_price, instrument,side)

            # trading service 会对精度和step兜底
            order_cmd = OrderCmd(
                instId=self.inst,
                side=side,
                ordType="limit",
                sz=str(target_size),
                px=str(target_price),
                tdMode="isolated",
                posSide="net",
                leverage=self.leverage
            )

            # TODO LLM Check
            logger.info("[ORDER CMD] {}".format(order_cmd))
            order_ack = await self.trading_service.submit_limit(order_cmd, await_live=False)
            logger.info("[ORDER ACK]".format(order_ack))
            
            if order_ack.accepted:
                self._pending_open_order_id = order_ack.ordId
                state_change_payload = {
                    "side": side,
                    "size": target_size,
                    "opened_ts": None
                }
                await self._transition_to(PositionState.OPENING, state_change_payload)
            else:
                logger.warning(f"Failed to submit order: {order_ack}")

            logger.debug(f"New Order submitted: {order_ack}")
        
        except Exception as e:
            logger.error(f"Error processing NO_POSITION action {action}: {e}")

    
    async def _handle_open_position(self, 
                                    action: str, # ["HOLD", "CLOSE"]
                                    ):

        if action.upper() == "RIDE_PROFIT":
            return
        if action.upper() == "CLOSE_SHORT" or action.upper() == "CLOSE_LONG":
            side = "sell" if self.postion_state.side.lower() == "buy" else "buy"
            try:
                order_cmd = OrderCmd(
                    instId=self.inst,
                    side=side,
                    ordType="market",
                    sz=str(self.postion_state.size),
                    tdMode="isolated",
                    posSide="net",
                )
                order_ack = await self.trading_service.submit_market(order_cmd, await_live=False)
                logger.info(f"Close Market Order submitted: {order_ack}")
                if order_ack.accepted:
                    self._pending_close_order_id = order_ack.ordId
                    await self._transition_to(PositionState.CLOSING)
                else:
                    logger.warning(f"Failed to submit order: {order_ack}")
                # self._on_close(CloseRequest(inst=self.inst, reason="TradeMachine Decision"))
            except Exception as e:
                logger.error(f"Error processing OPEN action CLOSE: {e}")

    def _calculate_order_size(self, 
                              available_balance_str: str, 
                              target_price: MarketTicker,
                              instrument: Instrument, 
                              side: str,
                              ) -> Optional[float]:
        try:
            available_balance = float(available_balance_str)    
            max_notional_value = available_balance * self.leverage
            
            reference_price = float(target_price.askPx) if side == "buy" else float(target_price.bidPx)
            reference_price = float(target_price.last)
            # max_coin_size = max_notional_value / reference_price
            # target_contract_size_raw = max_coin_size / instrument.ctVal
            # return target_contract_size_raw, reference_price
            max_contracts_raw = ((available_balance * 0.99) * self.leverage) / (instrument.ctVal * reference_price)
            return max_contracts_raw, reference_price
            
        except Exception as e:
            logger.error(f"Error calculating order size: {e}")
            return None, None

    # Binding in OrdersFeed
    async def on_order_event(self, evt: Dict[str, Any]):
        """
        evt: OrdersFeed 从 ws 下游拿到的单条 order 事件。
        类似 OKX:
        { 
            'instType': 'SWAP',  'instId': 'DOGE-USDT-SWAP', 'tgtCcy': '', 'ccy': 'USDT', 
            'tradeQuoteCcy': '', 'ordId': '3040123512394338304', 'clOrdId': '', 'algoClOrdId': '', 
            'algoId': '', 'tag': '', 'px': '', 'sz': '0.11', 'notionalUsd': '18.37217712', 
            'ordType': 'market', 'side': 'buy', 'posSide': 'net', 'tdMode': 'isolated', 
            'accFillSz': '0', 'fillNotionalUsd': '', 'avgPx': '0', 
            'state': 'live', 'lever': '5', 'pnl': '0', 'feeCcy': 'USDT', 'fee': '0',
        } 
        """
        
        inst_id = evt.get("instId")
        if inst_id and inst_id != self.inst:
            # 不是本 instrument 的单，忽略
            return

        ord_id = evt.get("ordId") or evt.get("clOrdId")
        if not ord_id:
            return

        state = evt.get("state") or evt.get("status")
        side  = evt.get("side")
        acc_fill_sz_str = evt.get("accFillSz")  # OKX 风格
        avg_px_str      = evt.get("avgPx")

        # ------------------------
        # 1) 处理开仓挂单（OPENING）
        # ------------------------
        if ord_id == self._pending_open_order_id:
            await self._handle_open_order_event(
                state=state,
                side=side,
                acc_fill_sz_str=acc_fill_sz_str,
                avg_px_str=avg_px_str,
                evt=evt,
            )
            return

        # ------------------------
        # 2) 处理平仓挂单（CLOSING）
        # ------------------------
        if ord_id == self._pending_close_order_id:
            await self._handle_close_order_event(
                state=state,
                side=side,
                acc_fill_sz_str=acc_fill_sz_str,
                avg_px_str=avg_px_str,
                evt=evt,
            )
            return
    
    async def _handle_open_order_event(
        self,
        state: str,
        side: Optional[str],
        acc_fill_sz_str: Optional[str],
        avg_px_str: Optional[str],
        evt: Dict[str, Any],
    ):
        logger.debug(f"[OPEN_ORDER_EVENT] state={state}, evt={evt}")

        FILLED_STATES = {"filled", "partially_filled"}
        CANCELED_STATES = {"canceled", "canceled_by_system"}
        REJECTED_STATES = {"rejected"}

        if state in FILLED_STATES:
            filled_sz = float(acc_fill_sz_str) if acc_fill_sz_str else float("0")
            state_change_payload = {
                "side": side or self.postion_state.side,
                "size": float(filled_sz) if filled_sz else self.postion_state.size,
                "opened_ts": time.time(),
            }
            await self._transition_to(PositionState.OPEN, state_change_payload)
            
        elif state in CANCELED_STATES | REJECTED_STATES:
            logger.warning(
                f"Open order failed or canceled, state={state}, evt={evt}"
            )
            # 回到空仓
            state_change_payload = {
                "side": None,
                "size": None,
                "opened_ts": None
            }
            await self._transition_to(PositionState.NO_POSITION, state_change_payload)
            self._pending_open_order_id = None
        else:
            logger.debug(f"Open order still pending, state={state}")
            
    async def _handle_close_order_event(
        self,
        state: str,
        side: Optional[str],
        acc_fill_sz_str: Optional[str],
        avg_px_str: Optional[str],
        evt: Dict[str, Any],
    ):
        logger.debug(f"[CLOSE_ORDER_EVENT] state={state}, evt={evt}")

        FILLED_STATES = {"filled"}
        CANCELED_STATES = {"canceled", "canceled_by_system"}
        REJECTED_STATES = {"rejected"}

        if state in FILLED_STATES:
            state_change_payload = {
                "side": None,
                "size": None,
                "closed_ts": time.time()
            }
            await self._transition_to(PositionState.NO_POSITION, state_change_payload)
            
            self._pending_close_order_id = None

        elif state in CANCELED_STATES | REJECTED_STATES:
            logger.warning(
                f"Close order not filled (state={state}), keep position OPEN, evt={evt}"
            )
            # 平仓失败：位置依然是 OPEN
            await self._transition_to(PositionState.OPEN)
            self._pending_close_order_id = None
        else:
            # live / partially_filled 等待继续成交
            logger.debug(f"Close order still pending, state={state}")
    