# utils/interaction_writer.py
import asyncio, json, time
from pathlib import Path
from typing import Any, Dict, Optional

class InteractionWriter:
    """
    轻量异步 JSONL 写入器：把 agent 的 prompt/response/metrics 以一行一JSON 的方式持久化。
    - 非阻塞：通过 asyncio.Queue 背景消费
    - 崩溃安全：每条 open-append-close，避免长时间持有句柄
    """
    def __init__(self, jsonl_path: str = "data/interactions.jsonl", max_queue: int = 2000):
        self.path = Path(jsonl_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.q: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self._stopping.set()
        if self._task:
            await self.q.put(None)  # 哨兵
            await self._task

    async def log(self, record: Dict[str, Any]):
        """将记录放入队列；如满则丢弃最老的1条以避免阻塞（可按需改成await）"""
        try:
            self.q.put_nowait(record)
        except asyncio.QueueFull:
            try:
                _ = self.q.get_nowait()
                self.q.task_done()
            except Exception:
                pass
            await self.q.put(record)

    async def _run(self):
        while True:
            item = await self.q.get()
            try:
                if item is None:
                    break
                line = json.dumps(item, ensure_ascii=False)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            finally:
                self.q.task_done()
        # drain 完成
