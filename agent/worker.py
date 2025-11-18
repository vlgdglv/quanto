# instrument_worker.py
import asyncio, time
from datetime import datetime

from collections import deque
from typing import Dict, Tuple, Optional, List

from agent.states import SharedState
from agent.tradings import PositionProvider, PositionSnapshot
from agent.schemas import FeatureFrame, RegimeSignal, DirectionSignal, TimingSignal, TradeIntent, TF, Side
from agent.llm_agents import (run_regime_agent, run_direction_agent,
                              run_flat_timing_agent, run_position_timing_agent,
                              build_agent_snapshot, format_position_for_prompt)

from utils.logger import logger
from trading.models import Position

class LatestStore:
    def __init__(self): self._m: Dict[Tuple[str,TF], FeatureFrame] = {}
    def update(self, f: FeatureFrame):
        self._m[(f.inst,f.tf)] = FeatureFrame(inst=f.inst, tf=f.tf, ts_close=f.ts_close, features=f.features, kind=f.kind)
    def get(self, inst: str, tf: TF) -> Optional[FeatureFrame]: return self._m.get((inst,tf))

class AgentState:
    def __init__(self):
        self.regime: Optional[RegimeSignal] = None
        self.direction: Optional[DirectionSignal] = None

class InstrumentWorker:
    def __init__(self, inst, latest, q_factory, shared_state, position_provider,
                 emit_signal, emit_intent, use_30m_confirm=False, account_service=None):
        self.inst = inst
        self.latest = latest
        self.q4h = q_factory(inst, "4H")
        self.q1h = q_factory(inst, "1H")
        self.q15 = q_factory(inst, "15m")
        self.q30 = q_factory(inst, "30m") if use_30m_confirm else None
        self.shared = shared_state
        self.pos_provider = position_provider
        self.emit_signal = emit_signal
        self.emit_intent = emit_intent
        self.use_30m_confirm = use_30m_confirm
        self.default_size_pct = 10.0

        self.account_service = account_service

    async def start(self):
        self.tasks = [
            asyncio.create_task(self._loop_rd()),
            asyncio.create_task(self._loop_15m()),
        ]
        # if self.q30:
        #     self.tasks.append(asyncio.create_task(self._loop_30m_cache()))


    async def stop(self):
        for t in self.tasks:
            t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def _loop_rd(self):
        logger.info(f"RD worker started for {self.inst}")

        last_ts = {"4H": None, "1H": None}

        async def _drain(q: asyncio.Queue):
            items = []
            while True:
                try:
                    items.append(q.get_nowait())
                except asyncio.QueueEmpty:
                    break
            return items

        while True:
            new4 = new1 = False

            f4_list = await _drain(self.q4h)
            f1_list = await _drain(self.q1h)

            for f4 in f4_list:
                self.latest.update(f4)
                new4 = True
            for f1 in f1_list:
                self.latest.update(f1)
                new1 = True

            if new4:
                snap4h = self.latest.get(self.inst, "4H")
                if snap4h:
                    ts4 = getattr(snap4h, "ts_close", None)
                    if ts4 != last_ts["4H"]:
                        try:
                            s4 = build_agent_snapshot(snap4h)
                            logger.debug(f"Prepare RD-Regime: {self.inst} 4H={s4}")
                            out4 = await run_regime_agent(self.inst, s4)  # -> {regime, regime_conf, risks?}
                            await self.shared.merge_rd(regime=out4.regime,
                                                    regime_conf=out4.regime_confidence,
                                                    invalidation=getattr(out4, "invalidation", None))
                            last_ts["4H"] = ts4
                            cur = self.shared.get_rd()
                            if cur:
                                await self.emit_signal(self.inst, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                                                    #    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(snap4h.ts_close/1000)),
                                                    snap4h.ts_close,
                                                    {"type": "RDState-4H", "value": cur})
                        except Exception as e:
                            logger.exception(f"[RD LOOP] regime agent failed for {self.inst}: {e}")

            if new1:
                snap1h = self.latest.get(self.inst, "1H")
                if snap1h:
                    ts1 = getattr(snap1h, "ts_close", None)
                    if ts1 != last_ts["1H"]:
                        try:
                            s1 = build_agent_snapshot(snap1h)
                            logger.debug(f"Prepare RD-Direction: {self.inst} 1H={s1}")
                            out1 = await run_direction_agent(self.inst, s1)  # -> {direction, dir_conf, conflicts?}
                            await self.shared.merge_rd(direction=out1.direction,
                                                    dir_conf=out1.direction_confidence,
                                                    invalidation=getattr(out1, "invalidation", None))
                            last_ts["1H"] = ts1
                            cur = self.shared.get_rd()
                            if cur:
                                await self.emit_signal(self.inst, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                    #    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(snap1h.ts_close/1000)),
                                                    snap1h.ts_close,
                                                    {"type": "RDState-1H", "value": cur})
                        except Exception as e:
                            logger.exception(f"[RD LOOP] direction agent failed for {self.inst}: {e}")

            await asyncio.sleep(10.0)  

    async def _loop_30m_cache(self):
        while True:
            f30 = await self.q30.get()
            self.latest.update(f30)

    async def _loop_15m(self):
        """唯一发单入口：15m 到达 -> wait RD -> Timing -> Position -> TradeIntent"""
        logger.info(f"15m worker started for {self.inst}")
        cooldown_sec = 600  # 10分钟冷却，避免过度交易（可调）
        last_intent_ts = 0.0
        while True:
            f15 = await self.q15.get()
            logger.debug(f"loop_15m: get frame: {self.inst}")
            self.latest.update(f15)

            if not await self.shared.wait_rd(timeout_sec=2.0):
                logger.debug("RD not ready for {}, reason={}".format(self.inst, self.shared.not_ready_reason()))
                continue
            
            rd = self.shared.get_rd()
            if rd is None:
                logger.debug(f"RD is None for {self.inst}")
                continue    
            
            try:
                snap15 = self.latest.get(self.inst, "15m")
                snap15 = build_agent_snapshot(snap15)
                # snap30 = self.latest.get(self.inst, "30m") if self.use_30m_confirm else None

                logger.debug(f"Prepare Timing: {self.inst} 15m={snap15}, rd={rd}")

                position: List[Position] = await self.account_service.get_positions(instId=self.inst)
                position = format_position_for_prompt(position, self.inst)

                timing_type = "Timing"
                if position is None or position == "NONE":
                    timing = await run_flat_timing_agent(self.inst, rd, snap15, None, position)
                    timing_type = "FlatTiming"
                else:
                    timing = await run_position_timing_agent(self.inst, rd, snap15, None, position)
                    timing_type = "PositionTiming"

                await self.emit_intent(self.inst, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                       f15.ts_close,
                                       {"type": timing_type, "value": timing.model_dump()})
            except Exception as e:
                logger.exception(f"[Timing LOOP] rd_agent step failed for {self.inst}: {e}")

            # NOT NOW
            # now = time.time()
            # if timing.action in ("ENTER","ADD") and now - last_intent_ts < cooldown_sec:
            #     continue
            


            # intent = trading_agent_build_intent(
            #     inst=self.inst, ts=f15.ts_close, rd=rd, timing=timing, pos=pos,
            #     default_size_pct=self.default_size_pct
            # )
            # if intent.action_norm != "SKIP":
            #     await self.emit_intent(self.inst, intent)
            #     last_intent_ts = now
            await asyncio.sleep(10)
            