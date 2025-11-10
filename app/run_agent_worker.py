# app/run_agent_worker.py
import os, time
import asyncio
from collections import defaultdict
from typing import Dict, Tuple
import csv
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.okx/.env"))
OKX_API_KEY = os.getenv("OKX_API_KEY")
OKX_SECRET = os.getenv("OKX_SECRET_KEY")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE")

os.environ["ALL_PROXY"] = os.getenv("ALL_PROXY", "http://127.0.0.1:7897")
for k in ("HTTP_PROXY","HTTPS_PROXY","http_proxy","https_proxy"):
    os.environ.pop(k, None)
for k in ("ALL_PROXY","all_proxy"):
    os.environ.pop(k, None)

for k in ("ALL_PROXY","all_proxy","HTTPS_PROXY","https_proxy","HTTP_PROXY","http_proxy"):
    v = os.getenv(k)
    if v:
        print(f"[proxy] {k}={v}")

from infra.redis_stream import RedisStreamsSubscriber

from agent.schemas import FeatureFrame, TF
from agent.worker import InstrumentWorker, LatestStore
from agent.states import SharedState
from agent.tradings import PositionSnapshot, PositionProvider
from utils import logger, load_cfg
from trading.services.endpoints import make_endpoints_from_cfg
from trading.services.account_service import AccountService

from infra import HttpContainer

_DEEP_BLUE  = "\033[34m"
_DEEP_GREEN = "\033[32m"
_BRIGHT_BLUE  = "\033[1;34m"
_BRIGHT_GREEN = "\033[1;32m"
_RESET      = "\033[0m"
AGENT_BASE_DIR = "data/agents"

_csv_lock = asyncio.Lock()

def _csv_path(base_dir: str | Path, dt: datetime | None = None, inst: str | None = None) -> Path:
    dt = dt or datetime.now()
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    # 文件名：agent_desicions_YYYYMMDD.csv
    return base / f"agent_desicions_{inst}_{dt.strftime('%Y%m%d')}.csv"

async def _append_csv_row(base_dir: str | Path, emit_time: str, tf: str, inst: str, obj) -> None:
    path = _csv_path(base_dir, inst=inst)
    row = [emit_time, tf, inst, str(obj)]
    async with _csv_lock:
        is_new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["ts", "tf", "inst", "content"])
            w.writerow(row)

async def emit_signal(inst: str, emit_time, obj, base_dir: str | Path = AGENT_BASE_DIR):
    print(f"{_BRIGHT_BLUE}[signal][{inst}][{emit_time}] {obj}{_RESET}")
    await _append_csv_row(base_dir, emit_time, "rd", inst, obj)

async def emit_intent(inst: str, emit_time, intent, base_dir: str | Path = AGENT_BASE_DIR):
    payload = intent.model_dump() if hasattr(intent, "model_dump") else intent
    print(f"{_BRIGHT_GREEN}[intent][{inst}][{emit_time}] {payload}{_RESET}")
    await _append_csv_row(base_dir, emit_time, "intent", inst, payload)


async def main():
    REDIS_DSN   = os.getenv("REDIS_DSN",   "redis://:12345678@127.0.0.1:6379/0")
    FEATURES_STREAM = os.getenv("FEATURES_STREAM", "DOGE-USDT-SWAP")
    FEATURES_START = os.getenv("FEATURES_START", "now")
    USE_30M_CONFIRM = bool(int(os.getenv("USE_30M_CONFIRM", False)))
    INSTS = [s.strip() for s in os.getenv("INSTS", "DOGE-USDT-SWAP").split(",") if s.strip()]

    x = 17
    x_min_ago_ms = int((time.time() - x * 60) * 1000)
    start_id = f"{x_min_ago_ms}-0"  
    FEATURES_START = start_id
    # FEATURES_START = "now"

    subscriber = RedisStreamsSubscriber(
        dsn=REDIS_DSN,
        stream=FEATURES_STREAM,
        start= FEATURES_START
    )

    latest = LatestStore()
    queues: Dict[Tuple[str, TF], asyncio.Queue] = defaultdict(lambda: asyncio.Queue(maxsize=2048))

    def queue_factory(inst: str, tf: TF) -> asyncio.Queue:
        return queues[(inst, tf)]

    pos_provider = PositionProvider()
    shared_states: Dict[str, SharedState] = {inst: SharedState(rd_ttl_sec=3600) for inst in INSTS}
    
    trading_cfg = load_cfg("configs/okx_trading_config.yaml")
    endpoints = make_endpoints_from_cfg(trading_cfg)

    container = await HttpContainer.start(trading_cfg, logger, time_sync_interval_sec=600)
    http = container.http
    account_service = AccountService(http, endpoints)

    workers = [
        InstrumentWorker(
            inst=inst,
            latest=latest,
            q_factory=queue_factory,
            shared_state=shared_states[inst],
            position_provider=pos_provider,
            emit_signal=emit_signal,
            emit_intent=emit_intent,
            use_30m_confirm=USE_30M_CONFIRM,
            account_service=account_service,
        )
        for inst in INSTS
    ]
    for w in workers:
        await w.start()

    async def on_message(payload: dict):
        try:
            f = FeatureFrame(**payload)
            latest.update(f)
            q = queues.get((f.inst, f.tf))
        except Exception:
            logger.warning(f"Bad payload dropped, payload={str(payload)[:200]}")
            return
        # print("[feature]", f.inst, f.tf, f.ts_close)
        
        if q:
            try:
                q.put_nowait(f)
            except asyncio.QueueFull:
                try:
                    _ = q.get_nowait()
                    q.put_nowait(f)
                except Exception as e:
                    logger.warning("Exception in on_message: %s", str(e))
            except Exception as e:
                logger.warning("Exception in on_message: %s", str(e))
                
    async def on_message_with_mock(payload: dict):
        # quick validation and canonicalization
        try:
            f = FeatureFrame(**payload)
            print("[feature]", f.inst, f.tf, f.ts_close)
        except Exception:
            # invalid payload: drop
            logger.warning(f"Bad payload dropped, payload={str(payload)[:200]}")
            return

        # helper to create a new frame with same content but overwritten tf and ts (if desired)
        def _mk_frame(orig: FeatureFrame, tf_new: str) -> FeatureFrame:
            # Keep same ts by default; you may want to adjust ts if you need distinct epochs
            new_payload = {
                "inst": orig.inst,
                "tf": tf_new,
                "ts_close": orig.ts_close,
                "features": orig.features,
                "kind": orig.kind
            }
            return FeatureFrame(**new_payload)

        incoming_tf = f.tf
        if incoming_tf == "30m":
            f = _mk_frame(f, "4H")
        elif incoming_tf == "15m":
            f = _mk_frame(f, "1H")
        elif incoming_tf == "5m":
            f = _mk_frame(f, "15m")
        else:
            return
        
        print("[MOCK] inst %s, tf (%s->%s), ts %s" % (f.inst, incoming_tf, f.tf, f.ts_close))

        latest.update(f)
        q = queues.get((f.inst, f.tf))
        if q:
            try:
                q.put_nowait(f)
            except asyncio.QueueFull:
                try:
                    _ = q.get_nowait()
                    q.put_nowait(f)
                except Exception as e:
                    logger.warning("Exception in on_message_with_mock: %s", str(e))
                

        # list of frames to push: include original
        # frames_to_push = [f]

        # # --- Mocking rules ---
        # incoming_tf = f.tf
        # # 30m -> also act as 4H and 1H (for testing RD)
        # if incoming_tf == "30m":
        #     frames_to_push.append(_mk_frame(f, "4H"))
        #     frames_to_push.append(_mk_frame(f, "1H"))
        # # 15m -> also act as 30m (so timing agent sees both)
        # elif incoming_tf == "15m":
        #     frames_to_push.append(_mk_frame(f, "30m"))
        # # 5m -> act as 15m
        # elif incoming_tf == "5m":
        #     frames_to_push.append(_mk_frame(f, "15m"))
        # # (optionally) if incoming is a specially marked test tf e.g., "mock_all",
        # # you can expand here to create all TFs.
        # else:
        #     return
        
        # # --- Push all frames into latest store and corresponding queues ---
        # for frame in frames_to_push:
        #     try:
        #         # update latest for each (keeps SharedStore / LatestStore consistent)
        #         latest.update(frame)
        #     except Exception as e:
        #         # ignore latest update failures for testing
        #         logger.warning(f"Failed to update latest {e}, payload={str(payload)[:200]}")
        #     print("[MOCK] %s %s %s" % (frame.inst, frame.tf, frame.ts_close))
        #     # route to queue if queue exists
        #     q = queues.get((frame.inst, frame.tf))
        #     if q:
        #         try:
        #             q.put_nowait(frame)
        #         except asyncio.QueueFull:
        #             # drop oldest, keep newest
        #             try:
        #                 _ = q.get_nowait()
        #                 q.put_nowait(frame)
        #             except Exception:
        #                 # if even that fails, skip silently in test
        #                 logger.warning(
        #                     f"Failed to push to queue, payload={str(payload)[:200]}"
        #                 )

    logger.info(
        "Agent app started insts: {}, redis: {}, stream: {}".format(
            INSTS, REDIS_DSN, FEATURES_STREAM
        )
    )

    try:
        await subscriber.run_forever(on_message)
    finally:
        for w in workers:
            await w.stop()

if __name__ == "__main__":
    asyncio.run(main())
