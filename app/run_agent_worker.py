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
# Please do not use live account!
# OKX_API_KEY = os.getenv("OKX_API_KEY")
# OKX_SECRET = os.getenv("OKX_SECRET_KEY")
# OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE")
OKX_API_KEY_PAPER = os.getenv("OKX_API_KEY_PAPER")
OKX_SECRET_PAPER = os.getenv("OKX_SECRET_KEY_PAPER")
OKX_PASSPHRASE_PAPER = os.getenv("OKX_PASSPHRASE_PAPER")

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
from infra.data_relay import DataRelay

from agent.schemas import FeatureFrame, TF
from agent.agent_orchestrator import InstrumentAgentOrchestrator
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

async def show_trend(inst: str, emit_time, frame_time,  obj, base_dir: str | Path = AGENT_BASE_DIR):
    print(f"{_BRIGHT_BLUE}[trend][{inst}][emit_time:{emit_time}][frame_time:{datetime.strptime(frame_time, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")}] {obj}{_RESET}")
    await _append_csv_row(base_dir, emit_time, "rd", inst, obj)

async def show_trigger(inst: str, emit_time, frame_time, intent,  base_dir: str | Path = AGENT_BASE_DIR):
    payload = intent.model_dump() if hasattr(intent, "model_dump") else intent
    print(f"{_BRIGHT_GREEN}[trigger][{inst}][emit_time:{emit_time}][frame_time:{datetime.strptime(frame_time, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")}] {payload}{_RESET}")
    await _append_csv_row(base_dir, emit_time, "intent", inst, payload)


async def main():
    REDIS_DSN   = os.getenv("REDIS_DSN",   "redis://:12345678@127.0.0.1:6379/0")
    FEATURES_STREAM = os.getenv("FEATURES_STREAM", "DOGE-USDT-SWAP")
    FEATURES_START = os.getenv("FEATURES_START", "now")
    # INSTS = [s.strip() for s in os.getenv("INSTS", "DOGE-USDT-SWAP").split(",") if s.strip()]
    INST = os.getenv("INSTS", "DOGE-USDT-SWAP")

    x = 4 * 60
    x_min_ago_ms = int((time.time() - x * 60) * 1000)
    start_id = f"{x_min_ago_ms}-0"  
    FEATURES_START = start_id
    logger.info(f"Starting at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(x_min_ago_ms)/1000))}")
    # FEATURES_START = "now"

    subscriber = RedisStreamsSubscriber(
        dsn=REDIS_DSN,
        stream=FEATURES_STREAM,
        start= FEATURES_START
    )

    trading_cfg = load_cfg("configs/okx_trading_config.yaml")
    endpoints = make_endpoints_from_cfg(trading_cfg)

    container = await HttpContainer.start(trading_cfg, logger, 
                                          api_key=OKX_API_KEY_PAPER,
                                          secret_key=OKX_SECRET_PAPER,
                                          passphrase=OKX_PASSPHRASE_PAPER,
                                          time_sync_interval_sec=600)
    account_service = AccountService(container.http, endpoints)

    data_relay = DataRelay()

    await data_relay.start(subscriber)
    # worker = InstrumentAgentOrchestrator(
    #     inst=INST,
    #     data_relay=data_relay,
    #     account_service=account_service,
    #     trade_machine=None,
    #     anchor_tf="30m",
    #     driver_tf="15m",
    #     trigger_tf="5m",
    #     trend_callback=show_trend,
    #     trigger_callback=show_trigger
    # )
    
    # await asyncio.gather(
    #     data_relay.start(subscriber),
    #     worker.start()
    # )

    logger.info(
        "Agent app started insts: {}, redis: {}, stream: {}".format(
            INST, REDIS_DSN, FEATURES_STREAM
        )
    )

if __name__ == "__main__":
    asyncio.run(main())
