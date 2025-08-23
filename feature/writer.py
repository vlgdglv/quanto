# feature/writer.py
from __future__ import annotations
import asyncio, time, contextlib
from typing import List, Optional
import pandas as pd
from feature.sinks import FeatureSink

from utils.logger import logger

class FeatureWriter:
    def __init__(self,
                 sinks: List[FeatureSink],
                 flush_interval_s: float = 5.0,
                 max_buffer_rows: int = 1000):
        self.sinks = sinks
        self.flush_interval_s = float(flush_interval_s)
        self.max_buffer_rows = int(max_buffer_rows)

        self._queue: Optional[asyncio.Queue[pd.DataFrame]] = None
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stopping = False
        
    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=4096) 
        self._task = asyncio.create_task(self._run())
        logger.info("Feature writer initialized and started.")

    def add(self, df: pd.DataFrame):
        if df is None or df.empty:
            return
        if not self._loop:
            return
        def _put():
            try:
                self._queue.put_nowait(df.copy())
            except asyncio.QueueFull:
                logger.warning("[FeatureWriter] queue full — dropping %d rows", len(df))
        self._loop.call_soon_threadsafe(_put)

    async def stop(self):
        self._stopping = True
        if self._task:
            await self._drain_and_flush()
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        for s in self.sinks:
            with contextlib.suppress(Exception):
                s.close()
    
    async def _drain_and_flush(self):
        bufs = []
        while not self._queue.empty():
            try:
                bufs.append(self._queue.get_nowait())
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        if bufs:
            await self._flush(pd.concat(bufs, ignore_index=True))

    async def _run(self):
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        buf: list[pd.DataFrame] = []
        rows = 0
        last = loop.time()

        try:
            while True:
                elapsed = loop.time() - last
                remain = self.flush_interval_s - elapsed
                if remain <= 0:
                    try:
                        df = self._queue.get_nowait()
                        buf.append(df); rows += len(df)
                        self._queue.task_done()
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.1)
                else:        
                    try:
                        df = await asyncio.wait_for(self._queue.get(), timeout=remain)
                        rows += len(df); buf.append(df)
                        self._queue.task_done()
                    except asyncio.TimeoutError:
                        pass
                while True:
                    try:
                        df_now = self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    rows += len(df_now); buf.append(df_now)
                    self._queue.task_done()

                if buf and ((loop.time() - last) >= self.flush_interval_s or rows >= self.max_buffer_rows):    
                    data = pd.concat(buf, ignore_index=True)
                    await self._flush(data)
                    buf.clear()
                    rows = 0
                    last = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[FeatureWriter] _run crashed")
            raise

    async def _flush(self, df: pd.DataFrame):
        logger.info("Flushing dataframe")
        for s in self.sinks:
            try:
                s.write(df)
            except Exception as e:
                logger.exception(f"[FeatureWriter] sink {type(s).__name__} write error: {e}")
    