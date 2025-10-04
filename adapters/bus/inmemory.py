import asyncio
from typing import Dict, Any, AsyncIterator, DefaultDict
from collections import defaultdict

class InMemoryBus:
    def __init__(self, maxsize: int = 2000):
        self._topics: DefaultDict[str, asyncio.Queue] = defaultdict(
            lambda: asyncio.Queue(maxsize=maxsize)
        )

    async def publish(self, topic: str, event: Dict[str, Any]) -> None:
        await self._topics[topic].put(event)

    async def subscribe(self, topic: str) -> AsyncIterator[Dict[str, Any]]:
        q = self._topics[topic]
        while True:
            ev = await q.get()
            try:
                yield ev
            finally:
                q.task_done()
