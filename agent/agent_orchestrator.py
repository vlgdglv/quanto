# instrument_worker.py
import asyncio, time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable

from agent.schemas import FeatureFrame, RegimeSignal, DirectionSignal, TF, Side
from agent.trade_machine import TradeMachine, PositionState
from agent.agent_hub import invoke_trend_agent, invoke_trigger_agent, invoke_entry_agent, invoke_exit_agent
from agent.agent_hub.trend_agent import TrendOutput


from infra.data_relay import DataRelay
from utils.logger import logger
from trading.models import Position
from trading.services.account_service import AccountService

@dataclass
class ContextSignal:
    payload: Any
    ts_updated: float = field(default_factory=time.time)
    valid_ttl: int = 4.35 * 3600

    @property
    def is_stale(self) -> bool:
        return (time.time() - self.ts_updated) > self.valid_ttl

tf2seconds = {
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1H": 3600,
    "4H": 14400,
}

class ContextBoard:
    def __init__(self, driver_tf="1H"):
        self._frames: Dict[str, FeatureFrame] = {}
        self._signal: Optional[ContextSignal] = None
        self._lock = asyncio.Lock()
        self.latest_driver_ts = None
        self.processed_driver_ts = None
        self.driver_tf = driver_tf
    
    def update_frame(self, f: FeatureFrame):
        self._frames[(f.inst, f.tf)] = f
        if f.tf == self.driver_tf:
            self.latest_driver_ts = f.ts_close

    def get_frame(self, inst: str, tf: str) -> Optional[FeatureFrame]:
        return self._frames.get((inst, tf))

    async def update_signal(self, payload: Any,source_ts: str = None):
        async with self._lock:
            self._signal = ContextSignal(payload)
            if source_ts:
                self.processed_driver_ts = source_ts

    def get_signal(self) -> Optional[ContextSignal]:
        return self._signal


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
        logger.info("Trend Layer (1H+4H) started...")

        drain_anchor_task = asyncio.create_task(self._drain_to_board(self.anchor_queue, "4H"))
        self.tasks.append(drain_anchor_task)
        
        while True:
            driver_frame: FeatureFrame = await self.driver_queue.get() 
            self.board.update_frame(driver_frame)

            anchor_frame: FeatureFrame = self.board.get_frame(self.inst, self.tfs["anchor"])
            
            if not anchor_frame:
                logger.debug("Waiting for initial 4H data...")
                continue 

            try:
                # out = await strategy_agent_chain.ainvoke(...)
                """
                    Invoke strategy agent
                """
                last_signal = self.board.get_signal()
                if not last_signal:
                    last_context = None
                else:
                    last_context = last_signal.payload

                trend_output = await invoke_trend_agent(anchor_frame, driver_frame, last_context)
                
                if self.trend_callback:
                    await self.trend_callback(driver_frame.inst, driver_frame.ts_close, trend_output)

                await self.board.update_signal(trend_output, source_ts=driver_frame.ts_close)
            except Exception as e:
                logger.error(f"Trend loop failed: {e}")

    async def _drain_to_board(self, queue, tf_name):
        while True:
            frame: FeatureFrame = await queue.get()
            self.board.update_frame(frame)
            logger.debug(f"Updated Board with {tf_name}")

    async def worker_seek_trigger_opportunities(self):
        logger.info(f"Trigger worker started for {self.inst}")
        
        while True:
            trigger_frame: FeatureFrame = await self.trigger_queue.get()
            if trigger_frame:
                await self._wait_for_alignment(trigger_frame.ts_close)
                # print(f"Got trigger frame for {trigger_frame.inst} at {trigger_frame.ts_close}")
            position: List[Position] = await self.account_service.get_positions(instId=self.inst)
            
            trend_singal = self.board.get_signal()
            if not trend_singal:
                logger.debug("No strategy signal yet...")
                continue
            if trend_singal.is_stale:
                logger.debug("This strategy signal is stale. Waiting for valid strategy signal...")
                continue

            try:
                """
                    Invoke trigger agent        
                """
                if self.trade_machine is not None:
                    if self.trade_machine.has_position():
                        logger.info(f"Position exists for {self.inst}")
                        last_trigger = self.trade_machine.get_last_trigger_output()
                        trigger_out = await invoke_exit_agent(trend_singal.payload, trigger_frame, position, last_trigger)
                    else:
                        trigger_out = await invoke_entry_agent(trend_singal.payload, trigger_frame)
                else:
                    trigger_out = await invoke_trigger_agent(trend_singal.payload, trigger_frame, position)
                
                close_time = datetime.strptime(trigger_frame.ts_close, "%Y%m%d%H%M%S") + timedelta(seconds=tf2seconds.get(self.tfs["trigger"]))
                emit_time = datetime.now()
                if emit_time - close_time > timedelta(minutes=3): 
                    logger.warning("Warming up agents, no trade step taken")
                else:
                    result = await self.trade_machine.step(trigger_out)
                
                if self.trigger_callback:
                    await self.trigger_callback(trigger_frame.inst, trigger_frame.ts_close, trigger_out)

            except Exception as e:
                logger.exception(f"Trigger loop failed: {e}")
        
    async def _wait_for_alignment(self, trigger_ts_str: str):
        trig_secs = tf2seconds.get(self.tfs["trigger"])
        driver_secs = tf2seconds.get(self.tfs["driver"])
        
        if not trig_secs or not driver_secs:
            logger.error("Unknown TF in TF_SECS config")
            return

        ts_dt = datetime.strptime(trigger_ts_str, "%Y%m%d%H%M%S")
        ts_unix = ts_dt.timestamp()

        trigger_end_unix = ts_unix + trig_secs

        if trigger_end_unix % driver_secs == 0:
            target_driver_unix = trigger_end_unix - driver_secs
            target_driver_ts = datetime.fromtimestamp(target_driver_unix).strftime("%Y%m%d%H%M%S")
            
            logger.debug(f"Alignment check: Trigger {trigger_ts_str} aligns with Driver {target_driver_ts}. Waiting...")

            wait_start = datetime.now()
            while True:
                if (datetime.now() - wait_start).seconds > 120:
                    logger.warning(f"Alignment Timeout: Proceeding without confirmed Trend for {target_driver_ts}")
                    break
                
                if self.board.processed_driver_ts and self.board.processed_driver_ts >= target_driver_ts:
                    logger.info(f"Alignment synced. Trend ready for {self.board.processed_driver_ts}")
                    break
                
                await asyncio.sleep(0.1)
    
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