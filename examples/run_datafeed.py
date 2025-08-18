# examples/run_datafeed.py
import asyncio, yaml
from pathlib import Path
from datafeed.pipeline import DataPipeline, WriteProcessor
from datafeed.storage import MemoryStore, DiskStore, CompositeStore

def load_cfg():
    with open(Path(__file__).resolve().parents[1] / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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
