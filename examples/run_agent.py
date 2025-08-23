# examples/run_agent.py
from dotenv import load_dotenv
import time
import random
import contextlib
import asyncio, yaml
from pathlib import Path
from typing import Dict, Any
from agent.the_agent import Agent
from agent.schema import ActionProposal
from feature.engine_pd import FeatureEnginePD
from datafeed.pipeline import FeatureEngineProcessor
from datafeed.pipeline import DataPipeline
from utils.logger import logger

# ===== Rate limit & jitter config (minimal) =====
MAX_RPS = 1.5            # 每秒最多请求数（按你账号上限的 70–85% 设置）
BASE_JITTER = (0.4, 1.2) # 每次调用前的随机抖动，单位秒
RETRY_MAX = 6            # 最大重试次数
RETRY_CAP = 30.0         # 单次退避上限（秒）

_last_call_mono = 0.0    # 记录上一次真正出手调用的时间（time.monotonic）

SNAPSHOT_QUEUE: asyncio.Queue = asyncio.Queue(maxsize=256)

load_dotenv()

def load_cfg():
    with open(Path(__file__).resolve().parents[1] / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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

async def snapshot_consumer(agent: Agent, gate: ColdStartGate):
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
                    proposal: ActionProposal = await asyncio.to_thread(_do_call)
                    _last_call_mono = time.monotonic()
                    print("PROPOSAL", proposal.model_dump())
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

            # proposal: ActionProposal = await asyncio.to_thread(agent.propose, snapshot)
            # print("PROPOSAL", proposal.model_dump())
        except Exception as e:
            print("Agent Error:", e)
        finally:
            SNAPSHOT_QUEUE.task_done()


async def main():
    cfg = load_cfg()

    cold_cfg = (cfg.get("cold_start") or {})
    gate = ColdStartGate(
        duration_sec=cold_cfg.get("duration_sec", 0),
        min_snapshots=cold_cfg.get("min_snapshots", 0),
    )

     # 1) 特征引擎
    engine = FeatureEnginePD()
    # （可选）冷启动回补，避免指标漂移
    # await asyncio.to_thread(warm_up_engine, engine, "BTC-USDT-SWAP", "1m", 300)

    # 2) LLM Agent
    agent = Agent(model="deepseek-chat", temperature=1.0)

    # 3) 处理器：把快照放入异步队列
    processor = FeatureEngineProcessor(cfg, engine, on_snapshot=make_on_snapshot())

    # 4) 消费者协程
    consumer_task = asyncio.create_task(snapshot_consumer(agent, gate))

    # 5) 数据管道（你的 WS 客户端）
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

if __name__ == "__main__":
    import contextlib, asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass