# instrument_worker.py
import asyncio, time
from datetime import datetime

from dataclasses import dataclass
from collections import deque
from typing import Dict, Tuple, Optional, List, Callable

from agent.states import SharedState
from agent.schemas import FeatureFrame, RegimeSignal, DirectionSignal, TF, Side
from agent.llm_agents import (run_regime_agent, run_direction_agent,
                              run_flat_timing_agent, run_position_timing_agent,
                              build_agent_snapshot, format_position_for_prompt)
from agent.trade_machine import TradeMachine, PositionState

from utils.logger import logger
from trading.models import Position
from trading.services.account_service import AccountService

@dataclass
class StrategySignal:
    regime: str           # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float
    ts_updated: float
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

    async def update_strategy(self, regime: str, conf: float):
        async with self._lock:
            self._strategy = StrategySignal(
                regime=regime,
                confidence=conf,
                ts_updated=time.time()
            )


class InstrumentAgentOrchestrator:
    def __init__(self, 
                 inst: str,  
                 q_factory: Callable, 
                 shared_state: SharedState,
                 emit_signal: Optional[Callable] = None, 
                 emit_intent: Optional[Callable] = None, 
                 use_30m_confirm: bool = False, 
                 account_service: Optional[AccountService] = None,
                 trade_machine: Optional[TradeMachine] = None
                 ):
        self.inst = inst
        self.board = ContextBoard()
        self.q4h = q_factory(inst, "4H")
        self.q1h = q_factory(inst, "1H")
        self.q15 = q_factory(inst, "15m")
        self.q30 = q_factory(inst, "30m") if use_30m_confirm else None
        
        self.shared = shared_state
        self.emit_signal = emit_signal
        self.emit_intent = emit_intent
        self.use_30m_confirm = use_30m_confirm
        self.default_size_pct = 10.0

        self.account_service = account_service
        self.trade_machine = trade_machine
        if self.trade_machine is None:
            logger.info(f"No trading machine for {inst}")


    async def start(self):
        self.tasks = [
            asyncio.create_task(self.worker_maintain_strategic_context()),
            asyncio.create_task(self.worker_seek_tactical_opportunities()),
        ]
    
    async def stop(self):
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
    
    async def worker_maintain_strategic_context(self):
        logger.info("Strategy Layer (1H+4H) started...")

        drain_4h_task = asyncio.create_task(self._drain_to_board(self.q4h, "4H"))
        self.tasks.append(drain_4h_task)
        
        while True:
            f1h = await self.q1h.get() 
            self.board.update_frame(f1h)

            f4h = self.board.get_frame(self.inst, "4H")
            
            if not f4h:
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

    async def worker_seek_tactical_opportunities(self):
        logger.info(f"Signal worker started for {self.inst}")
        
        while True:
            f15 = await self.q15.get()
            self.latest.update(f15)
            
            # Basic risk management
            if self._is_emotional_tilt():
                logger.warning(f"STOP TRADING: Too many recent losses for {self.inst}")
                continue

            rd_state = self.shared.get_rd_nowait() 
            if self._is_rd_stale(rd_state):
                logger.warning("RD state is stale or missing, skipping 15m signal")
                continue

            try:
                snap15 = build_agent_snapshot(self.latest.get(self.inst, "15m"))

                """
                    Invoke tactical agent        
                """
                # timing = await run_position_timing_agent(
                #     self.inst, 
                #     rd_context=rd_state,
                #     snapshot=snap15, 
                #     positions=self._get_current_pos()
                # )

                result = await self.trade_machine.step(...)
                                        
            except Exception as e:
                logger.exception(f"15m Loop failed: {e}")
                
    def _is_rd_stale(self, rd_state):
        if not rd_state: return True
        return (time.time() - rd_state.get('calc_ts', 0)) > (4 * 3600 * 2)

    def _is_emotional_tilt(self):
        """MVP 风控：如果最近 3 笔全是亏损，且总亏损超过 5%，暂停交易"""
        if len(self.trade_history) < 3: return False
        
        last_3 = list(self.trade_history)[-3:]
        # 假设 history 里存的是 pnl_pct (例如 -0.02 代表亏 2%)
        losses = [t['pnl'] for t in last_3 if t['pnl'] < 0]
        
        if len(losses) == 3: # 连亏3笔
             total_loss = sum(losses)
             if total_loss < -0.05: # 累计亏损超过 5%
                 return True
        return False

    def record_trade_result(self, pnl_pct):
        self.trade_history.append({
            'ts': time.time(),
            'pnl': pnl_pct
        })