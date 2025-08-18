# examples/run_features.py
import asyncio, yaml
from pathlib import Path
from datafeed.pipeline import DataPipeline, FeatureEngineProcessor
from feature.engine_pd import FeatureEnginePD
from feature.integrate import process_msg


def load_cfg():
    with open(Path(__file__).resolve().parents[1] / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
async def main():
    cfg = load_cfg()
    engine = FeatureEnginePD()
    processor = FeatureEngineProcessor(cfg, engine)
    pipe = DataPipeline(cfg, processor=processor)
    try:
        await pipe.run()
    except KeyboardInterrupt:
        await pipe.stop()


if __name__ == "__main__":
    asyncio.run(main())
