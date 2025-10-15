# examples/run_agent.py
from dotenv import load_dotenv
import time
import json
import random
import contextlib
import asyncio, yaml
from pathlib import Path
from typing import Dict, Any

from agent.agent_house import Agent
from agent.chat_models import llm_factory
from agent.schema import ActionProposal
from agent.bb_agents import LLMDecider, Decision

from feature.engine_pd import FeatureEnginePD
from feature.processor import FeatureEngineProcessor
from datafeed.pipeline import DataPipeline
from utils.logger import logger
from utils.config import load_cfg

from feature.writer import FeatureWriter
from feature.sinks import CSVFeatureSink
# examples/run_agent.py （只展示新增/修改段）
from agent.interaction_writer import InteractionWriter

from pathlib import Path


def _write_jsonl(path: str, data: dict):
    """同步追加一行 JSON；由 append_jsonl 在线程池里调用，避免阻塞事件循环"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

async def append_jsonl(path: str, data: dict):
    """异步包装：在线程池里执行同步写入"""
    await asyncio.to_thread(_write_jsonl, path, data)


# ===== Rate limit & jitter config (minimal) =====
MAX_RPS = 1.5            # 每秒最多请求数（按你账号上限的 70–85% 设置）
BASE_JITTER = (0.4, 1.2) # 每次调用前的随机抖动，单位秒
RETRY_MAX = 6            # 最大重试次数
RETRY_CAP = 30.0         # 单次退避上限（秒）

_last_call_mono = 0.0    # 记录上一次真正出手调用的时间（time.monotonic）

SNAPSHOT_QUEUE: asyncio.Queue = asyncio.Queue(maxsize=256)

load_dotenv()

class ColdStartGate:
    def __init__(self, duration_sec: int = 0, min_snapshots: int = 0):
        self.duration_sec = max(0, int(duration_sec or 0))
        self.min_snapshots = max(0, int(min_snapshots or 0))
        self._start_monotonic = time.monotonic()
        self._count = 0
        self._event = asyncio.Event()
        if self.duration_sec == 0 and self.min_snapshots == 0:
            self._event.set()

    def note_snapshot(self):
        if not self._event.is_set():
            self._count += 1
            if self._ready_by_count() or self._ready_by_time():
                self._event.set()
                logger.info("Warmed up!")

    def _ready_by_count(self) -> bool:
        return (self.min_snapshots > 0) and (self._count >= self.min_snapshots)
    
    def _ready_by_time(self) -> bool:
        if self.duration_sec <= 0:
            return False
        return (time.monotonic() - self._start_monotonic) >= self.duration_sec
    
    def is_ready(self) -> bool:
        return self._event.is_set()
    
    async def wait_ready(self):
        await self._event.wait()


def make_on_snapshot():
    """
    返回给 FeatureEngineProcessor 的回调：
    把 snapshot 非阻塞地塞进 asyncio 队列，让独立消费者处理。
    """
    loop = asyncio.get_running_loop()

    def _cb(snapshot: Dict[str, Any]):
        loop.call_soon_threadsafe(SNAPSHOT_QUEUE.put_nowait, snapshot)
    return _cb

async def snapshot_consumer(agent: Agent, gate: ColdStartGate, interaction_path: str):
    """
    独立协程：消费 snapshot 队列，调用 LLM 产出 proposal。
    放在线程池中执行以免阻塞事件循环。
    """
    announced = False
    while True:
        snapshot = await SNAPSHOT_QUEUE.get()
        try:
            gate.note_snapshot()

            if not gate.is_ready():
                if not announced:
                    logger.info("[ColdStart] warming... skipping agent decisions")
                    announced = True
                continue
            
            if announced:
                logger.info("[ColdStart] done. agent decisions enabled.")
                announced = False

            # --- 1) 轻微抖动，避免整点扎堆 ---
            await asyncio.sleep(random.uniform(*BASE_JITTER))

            # --- 2) 简易集中限速（最小间隔法） ---
            global _last_call_mono
            min_interval = 1.0 / MAX_RPS
            now = time.monotonic()
            wait = max(0.0, min_interval - (now - _last_call_mono))
            if wait > 0:
                await asyncio.sleep(wait)
            
            # --- 3) 指数退避（仅识别数字型 Retry-After），在后台线程执行同步 propose ---
            def _do_call():
                return agent.propose(snapshot)

            delay = 1.0
            last_exc = None
            for _ in range(RETRY_MAX):
                try:
                    # proposal: ActionProposal = await asyncio.to_thread(_do_call)
                    decision: Decision = await asyncio.to_thread(_do_call)
                    _last_call_mono = time.monotonic()
                    print("PROPOSAL", decision)
                    try:
                        rec = {
                            "t": int(time.time() * 1000),
                            "tf": snapshot.get("tf"),
                            "instId": snapshot.get("instId"),
                            "snap_ts": snapshot.get("ts"),
                            "kind": "ok",
                            "proposal": decision,
                            "snapshot": snapshot,
                            }
                        await append_jsonl(interaction_path, rec)
                    except Exception as _:
                        pass
                    break
                except Exception as e:
                    last_exc = e

                    # 尝试从异常对象里拿数字型 Retry-After（若无则走指数退避）
                    retry_after = None
                    try:
                        resp = getattr(e, "response", None)
                        headers = getattr(resp, "headers", None)
                        if headers:
                            ra = headers.get("Retry-After") or headers.get("retry-after")
                            if ra and ra.isdigit():
                                retry_after = float(ra)
                                logger.info("Server told us to retry after {} s".format(retry_after))
                    except Exception:
                        pass

                    sleep_s = retry_after if (retry_after is not None) else min(delay, RETRY_CAP) + random.uniform(0, 0.25)
                    await asyncio.sleep(sleep_s)
                    delay = min(delay * 2, RETRY_CAP)
            else:
                raise RuntimeError(f"Agent call failed after retries: {last_exc}") from last_exc

        except Exception as e:
            print("Agent Error:", e)
            try:
                rec = {
                    "t": int(time.time() * 1000),
                    "tf": snapshot.get("tf"),
                    "instId": snapshot.get("instId"),
                    "snap_ts": snapshot.get("ts"),
                    "snapshot": snapshot,
                    "error": str(e),
                }
                await append_jsonl(interaction_path, rec)
            except Exception as _:
                pass
        finally:
            SNAPSHOT_QUEUE.task_done()


async def main():
    cfg = load_cfg()

    # === 冷启动门限 ===
    cold_cfg = (cfg.get("cold_start") or {})
    persist_cfg = (cfg or {}).get("persist", {})
    interactions_path = (persist_cfg.get("interactions_path") or "data/interactions.jsonl")

    gate = ColdStartGate(
        duration_sec=cold_cfg.get("duration_sec", 0),
        min_snapshots=cold_cfg.get("min_snapshots", 0),
    )

    # === Feature Writer 管线（与 run_features 对齐） ===
    persist_cfg = (cfg or {}).get("persist", {})
    csv_path   = persist_cfg.get("csv_path", "data/features.csv")
    flush_s    = float(persist_cfg.get("flush_interval_s", 5))
    max_rows   = int(persist_cfg.get("max_buffer_rows", 1000))

    sinks = [CSVFeatureSink(csv_path, by=["instId", "tf"])]
    writer = FeatureWriter(sinks, flush_interval_s=flush_s, max_buffer_rows=max_rows)
    await writer.start()


    try:
        engine = FeatureEnginePD(enable_summary=True)
        agent = LLMDecider(model="gpt-4o", temperature=0.3, mode="single")
        processor = FeatureEngineProcessor(cfg, engine, on_snapshot=make_on_snapshot(), feature_writer=writer)
        consumer_task = asyncio.create_task(snapshot_consumer(agent, gate, interactions_path))
        pipe = DataPipeline(cfg, processor=processor)

        try:
            await pipe.run()
        except asyncio.CancelledError:
            raise
        except KeyboardInterrupt:
            pass
        finally:
            await pipe.stop()
            consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer_task
    finally:
        await writer.stop()


if __name__ == "__main__":
    import contextlib, asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass