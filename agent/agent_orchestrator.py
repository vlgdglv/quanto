# instrument_worker.py
import asyncio, time
from datetime import datetime

from dataclasses import dataclass
from collections import deque
from typing import Dict, Tuple, Optional, List, Callable


from agent.schemas import FeatureFrame, RegimeSignal, DirectionSignal, TF, Side
from agent.trade_machine import TradeMachine, PositionState

from infra.data_relay import DataRelay
from utils.logger import logger
from trading.models import Position
from trading.services.account_service import AccountService


@dataclass
class StrategySignal:
    regime: str           # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float
    ts_updated: float
    justification: str = ""
    valid_ttl: int = 4 * 3600

    @property
    def is_stale(self) -> bool:
        return (time.time() - self.ts_updated) > self.valid_ttl


class ContextBoard:
    def __init__(self):
        self._frames: Dict[str, FeatureFrame] = {}
        self._strategy: Optional[StrategySignal] = None
        self._lock = asyncio.Lock()
    
    def update_frame(self, f: FeatureFrame):
        self._frames[(f.inst, f.tf)] = f

    def get_frame(self, inst: str, tf: str) -> Optional[FeatureFrame]:
        return self._frames.get((inst, tf))

    async def update_strategy(self, regime: str, conf: float, justification: str):
        async with self._lock:
            self._strategy = StrategySignal(
                regime=regime,
                confidence=conf,
                justification=justification,
                ts_updated=time.time()
            )
   
    def get_strategy(self) -> Optional[StrategySignal]:
        return self._strategy


class InstrumentAgentOrchestrator:
    def __init__(self, 
                 inst: str,  
                 data_relay: DataRelay,
                 account_service: Optional[AccountService] = None,
                 trade_machine: Optional[TradeMachine] = None,
                 anchor_tf: str = "4H",
                 driver_tf: str = "1H",
                 trigger_tf: str = "15m",
                 trend_callback: Optional[Callable] = None, 
                 trigger_callback: Optional[Callable] = None, 
                 ):
        self.inst = inst
        self.board = ContextBoard()
        self.anchor_queue = data_relay.subscribe(inst, anchor_tf, max_len=4)
        self.driver_queue = data_relay.subscribe(inst, driver_tf, max_len=10)
        self.trigger_queue = data_relay.subscribe(inst, trigger_tf, max_len=200)
        
        self.trend_callback = trend_callback
        self.trigger_callback = trigger_callback
        self.default_size_pct = 10.0

        self.account_service = account_service
        self.trade_machine = trade_machine
        
        self.tfs = {
            "anchor": anchor_tf,
            "driver": driver_tf,
            "trigger": trigger_tf
        }
        
        if self.trade_machine is None:
            logger.warning(f"No trading machine for {inst}, dry run agents")


    async def start(self):
        self.tasks = [
            asyncio.create_task(self.worker_maintain_trend_context()),
            asyncio.create_task(self.worker_seek_trigger_opportunities()),
        ]
    
    async def stop(self):
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
    
    async def worker_maintain_trend_context(self):
        logger.info("Strategy Layer (1H+4H) started...")

        drain_anchor_task = asyncio.create_task(self._drain_to_board(self.anchor_queue, "4H"))
        self.tasks.append(drain_anchor_task)
        
        while True:
            driver_frame = await self.driver_queue.get() 
            self.board.update_frame(driver_frame)

            anchor_frame = self.board.get_frame(self.inst, self.tfs["anchor"])
            
            if not anchor_frame:
                logger.debug("Waiting for initial 4H data...")
                continue 

            try:
                # out = await strategy_agent_chain.ainvoke(...)
                """
                    Invoke strategy agent
                """
                await self.board.update_strategy(...)
            except Exception as e:
                logger.error(f"Strategy Agent failed: {e}")

    async def _drain_to_board(self, queue, tf_name):
        while True:
            frame = await queue.get()
            self.board.update_frame(frame)
            logger.debug(f"Updated Board with {tf_name}")

    async def worker_seek_trigger_opportunities(self):
        logger.info(f"Signal worker started for {self.inst}")
        
        while True:
            trigger_frame = await self.trigger_queue.get()
            self.board.update_frame(trigger_frame)
            
            # Basic risk management here:
            # But no, no risk full push
            # if self._is_emotional_tilt():
            #     logger.warning(f"STOP TRADING: Too many recent losses for {self.inst}")
            #     continue

            anchor_singal: StrategySignal = self.board.get_strategy()
            if not anchor_singal:
                logger.debug("No strategy signal yet...")
                continue
            if anchor_singal.is_stale:
                logger.debug("This strategy signal is stale. Waiting for valid strategy signal...")
                continue

            try:
                """
                    Invoke trigger agent        
                """
                # snap15 = build_agent_snapshot(trigger_frame)
                result = await self.trade_machine.step(...)
                                 
            except Exception as e:
                logger.exception(f"15m Loop failed: {e}")
                
    def _is_rd_stale(self, rd_state):
        if not rd_state: return True
        return (time.time() - rd_state.get('calc_ts', 0)) > (4 * 3600 * 2)

    def _is_emotional_tilt(self):
        if len(self.trade_history) < 3: return False
        
        last_3 = list(self.trade_history)[-3:]
        losses = [t['pnl'] for t in last_3 if t['pnl'] < 0]
        
        if len(losses) == 3:
             total_loss = sum(losses)
             if total_loss < -0.0500:
                 return True
        return False

    def record_trade_result(self, pnl_pct):
        self.trade_history.append({
            'ts': time.time(),
            'pnl': pnl_pct
        })