# tests/smoke_publish.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os, asyncio, time, json, random
from infra.redis_stream import RedisStreamsPublisher

REDIS_DSN   = "redis://:12345678@127.0.0.1:6379/0"
STREAM_NAME = "features"

async def main():
    pub = RedisStreamsPublisher(dsn=REDIS_DSN, stream=STREAM_NAME)
    insts = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "DOGE-USDT-SWAP"]
    tfs   = ["1m", "5m", "15m"]

    now = int(time.time()*1000)
    tasks = []
    i = 0
    while(True):
        inst = random.choice(insts)
        tf   = random.choice(tfs)
        payload = {
            "kind": "FeaturesUpdated",
            "inst": inst,
            "tf": tf,
            "ts_close": (now // 60000) * 60000,
            "features": {"ema_12": round(random.random()*100, 4), "kdj_k": round(random.random()*100, 2)},
            "feature_version": "v1.0.0",
            "engine_id": "smoke-producer",
            "computed_at": now,
            "trace_id": f"smoke-{i}",
            "seq": i,
        }
        tasks.append(pub.publish(STREAM_NAME, payload))
        await pub.publish(STREAM_NAME, payload)
        await asyncio.sleep(5)
        i += 1

    await asyncio.gather(*tasks)
    print(f"Published {len(tasks)} messages to stream '{STREAM_NAME}'")

if __name__ == "__main__":
    asyncio.run(main())
