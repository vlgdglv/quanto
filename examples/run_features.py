# examples/run_features.py
import asyncio, yaml
from pathlib import Path
from datafeed.pipeline import DataPipeline
from feature.processor import FeatureEngineProcessor
from feature.engine_pd import FeatureEnginePD
from feature.writer import FeatureWriter
from feature.sinks import CSVFeatureSink


def load_cfg():
    with open(Path(__file__).resolve().parents[1] / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
async def main():
    cfg = load_cfg()
    persist_cfg = (cfg or {}).get("persist", {})
    csv_path   = persist_cfg.get("csv_path", "data/features.csv")
    # sqlite_db  = persist_cfg.get("sqlite_path", "data/features.db")
    # table_name = persist_cfg.get("sqlite_table", "features")
    flush_s    = float(persist_cfg.get("flush_interval_s", 5))
    max_rows   = int(persist_cfg.get("max_buffer_rows", 1000))
    sinks = [
        CSVFeatureSink(csv_path, by="instId"),
        # SQLiteFeatureSink(sqlite_db, table=table_name, columns=FeatureEnginePD.columns())
    ]
    writer = FeatureWriter(sinks, flush_interval_s=flush_s, max_buffer_rows=max_rows)
    await writer.start()

    engine = FeatureEnginePD(enable_summary=True)
    processor = FeatureEngineProcessor(cfg, engine, feature_writer=writer)
    pipe = DataPipeline(cfg, processor=processor)
    try:
        await pipe.run()
    except KeyboardInterrupt:
        await pipe.stop()
    finally:
        await writer.stop()


if __name__ == "__main__":
    asyncio.run(main())
