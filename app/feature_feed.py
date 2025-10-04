# apps/feature_feed.py
import os, asyncio
from infra.publisher import RedisStreamsPublisher
from feature.engine_pd import FeatureEnginePD
from feature.processor import FeatureEngineProcessor
from datafeed.pipeline import DataPipeline
from utils.config import load_cfg
from utils.logger import logger

async def main():
    cfg = load_cfg()

    # 1) Redis Publisher 注入
    redis_dsn = os.environ.get("REDIS_DSN", cfg.get("redis", {}).get("dsn", "redis://127.0.0.1:6379/0"))
    stream    = os.environ.get("REDIS_STREAM", cfg.get("redis", {}).get("stream", "features"))
    pub = RedisStreamsPublisher(dsn=redis_dsn, stream=stream, maxlen_approx=cfg.get("redis", {}).get("maxlen", 1_000_000))

    # 2) Feature 引擎 + Processor
    engine = FeatureEnginePD(enable_summary=True)
    processor = FeatureEngineProcessor(cfg, engine, publisher=pub, stream_name=stream)

    # 3) 数据管道（OKX WS -> processor.handle），队列在 DataPipeline 里
    pipe = DataPipeline(cfg, processor=processor)

    logger.info("Feature feed app starting… (Redis: %s, stream: %s)", redis_dsn, stream)
    try:
        await pipe.run()
    except asyncio.CancelledError:
        pass
    finally:
        await pipe.stop()

if __name__ == "__main__":
    asyncio.run(main())
