from typing import Any, Dict, Callable
from adapters.bus.inmemory import InMemoryBus

class FeatureWorker:
    def __init__(self, 
                 bus: InMemoryBus, 
                 build_snapshot_from_row: Callable[[Dict[str, Any]], Dict[str, Any]],
                 topic_out: str = "snapshots"
                 ):
        self.bus = bus
        self.build_snapshot_from_row = build_snapshot_from_row
        self.topic_out = topic_out

    async def on_rows(self, rows: list[Dict[str, Any]]):
        for row in rows:
            snap = self.build_snapshot_from_row(row)
            await self.bus.publish(self.topic_out, snap)
