# test/tail_features.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import asyncio
import json
from infra.redis_stream import RedisStreamsSubscriber  # ← 用刚才的订阅器

REDIS_DSN   = os.getenv("REDIS_DSN",   "redis://:12345678@127.0.0.1:6379/0")
STREAM_NAME = os.getenv("REDIS_STREAM","BTC-USDT-SWAP")
START_POS   = os.getenv("START", "now")
CONCURRENCY = int(os.getenv("CONCURRENCY", "1"))

def _pretty(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)

async def on_message(payload: dict):
    meta = {
        "kind":            payload.get("kind"),
        "inst":            payload.get("inst"),
        "tf":              payload.get("tf"),
        "ts_close":        payload.get("ts_close") or payload.get("ts"),
        "feature_version": payload.get("feature_version"),
        "engine_id":       payload.get("engine_id"),
        "computed_at":     payload.get("computed_at"),
        "trace_id":        payload.get("trace_id"),
        "seq":             payload.get("seq"),
    }
    features = payload.get("features") or {}
    print("\n=== Feature Snapshot ===")
    print("[meta]")
    print(_pretty(meta))
    # print("[features]")
    # print(features)

async def main():
    sub = RedisStreamsSubscriber(
        dsn=REDIS_DSN,
        stream=STREAM_NAME,
        start=START_POS,
        block_ms=3000,
        fetch_count=128,
        concurrency=CONCURRENCY,
    )
    print(f"Tailing stream '{STREAM_NAME}' from {START_POS.upper()} ... Ctrl-C to exit.")
    await sub.run_forever(on_message)

if __name__ == "__main__":
    asyncio.run(main())
