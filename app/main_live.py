import asyncio, yaml
from pathlib import Path

from adapters.bus.inmemory import InMemoryBus
from application.services.feature_worker import FeatureWorker
from application.services.agent_mesh_worker import AgentMeshWorker, AgentRouter

from feature.engine_pd import FeatureEnginePD
from feature.processor import FeatureEngineProcessor
from datafeed.pipeline import DataPipeline
from agent.the_agent import Agent as OriginalAgent
from adapters.agents.simple import SimpleAgentAdapter
# from adapters.agents.langchain import LangChainAgentAdapter  # 以后接入

from utils.logger import logger
from feature.integrate import build_snapshot_from_row 

async def main(cfg: dict):
    bus = InMemoryBus(maxsize=2000)

    # ===== Feature =====
    engine = FeatureEnginePD(enable_summary=True)
    processor = FeatureEngineProcessor(cfg, engine, on_snapshot=None)
    feat_worker = FeatureWorker(bus, build_snapshot_from_row)

    async def on_rows(rows):
        await feat_worker.on_rows(rows)
    processor.on_rows_async = on_rows

    base_agent = OriginalAgent(model="gpt-4o", temperature=0.0)
    simple_port = SimpleAgentAdapter(base_agent)

    # 预留 LangChain 集群位：
    # langchain_chain = build_langchain_cluster(cfg)     # 你后面实现
    # lcg_port = LangChainAgentAdapter(langchain_chain, input_key="snapshot")
    router = AgentRouter(primaries=[simple_port], fallbacks=[])  # 以后 primaries=[lcg_port, simple_port]

    mesh = AgentMeshWorker(bus, router, max_concurrency=2, buffer_size=200)

    # ===== DataFeed =====
    pipe = DataPipeline(cfg, processor=processor)

    logger.info("Booting live system: datafeed -> features -> snapshots(bus) -> agents(router) -> proposals(bus)")
    await asyncio.gather(
        asyncio.create_task(mesh.run()),
        asyncio.create_task(pipe.run()),
    )

if __name__ == "__main__":
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    asyncio.run(main(cfg))
