import asyncio, time, hashlib, json
from typing import Dict, Any, List, Callable, Optional
from adapters.bus.inmemory import InMemoryBus
from domain.interfaces import AgentPort

class AgentRouter:
    def __init__(self, primaries: List[AgentPort], fallbacks: Optional[List[AgentPort]] = None):
        self.primaries = primaries
        self.fallbacks = fallbacks or []

    async def propose(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        last_err = None
        for ag in self.primaries + self.fallbacks:
            try:
                return await ag.propose(snapshot)
            except Exception as e:
                last_err = e
                continue
        raise last_err or RuntimeError("no agent available")

class AgentMeshWorker:
    """
    订阅 snapshot 主题 → 调用 AgentRouter → 发布 proposal。
    内置有界 buffer 和“丢旧保新”的削峰策略。
    """
    def __init__(self, bus: InMemoryBus, router: AgentRouter,
                 topic_in="snapshots", topic_out="proposals",
                 max_concurrency: int = 2, buffer_size: int = 200):
        self.bus = bus
        self.router = router
        self.topic_in = topic_in
        self.topic_out = topic_out
        self.sem = asyncio.Semaphore(max_concurrency)
        self.buffer = asyncio.Queue(maxsize=buffer_size)

    async def run(self):
        async def producer():
            async for ev in self.bus.subscribe(self.topic_in):
                try:
                    self.buffer.put_nowait(ev)
                except asyncio.QueueFull:
                    _ = await self.buffer.get(); self.buffer.task_done()
                    await self.buffer.put(ev)

        async def consumer():
            while True:
                snap = await self.buffer.get()
                async with self.sem:
                    try:
                        prop = await self.router.propose(snap)
                        # 统一补齐元信息（若 agent 未写）
                        prop.setdefault("schema", "proposal.v1")
                        prop.setdefault("inst_id", snap.get("inst_id"))
                        prop.setdefault("tf", snap.get("tf"))
                        prop.setdefault("ts_decision", int(time.time()))
                        if "idempotency_key" not in prop:
                            prop["idempotency_key"] = hashlib.md5(
                                json.dumps(prop, sort_keys=True).encode()
                            ).hexdigest()
                        await self.bus.publish(self.topic_out, prop)
                    finally:
                        self.buffer.task_done()

        await asyncio.gather(producer(), consumer())
