# examples/run_datafeed.py
import asyncio, yaml
from pathlib import Path
from datafeed.pipeline import DataPipeline, WriteProcessor
from datafeed.storage import MemoryStore, DiskStore, CompositeStore
from utils.logger import logger
from utils.config import load_cfg

async def main():
    cfg = load_cfg()
    mem = MemoryStore()
    disk = DiskStore(cfg["datafeed"]["out_dir"], cfg["datafeed"].get("backend", "csv"))
    store = CompositeStore(mem, disk)
    write_processor = WriteProcessor(cfg, store)
    pipe = DataPipeline(cfg, write_processor)
    try:
        await pipe.run()
    except KeyboardInterrupt:
        await pipe.stop()


if __name__ == "__main__":
    asyncio.run(main())
