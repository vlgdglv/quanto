# infra/data_relay.py
import asyncio
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from agent.schemas import FeatureFrame
from utils import logger


class RollingQueue(asyncio.Queue):
    def put_nowait(self, item):
        if self.full():
            try:
                self.get_nowait()
            except asyncio.QueueEmpty:
                pass
        super().put_nowait(item)
        

class DataRelay:
    def __init__(self):
        self._subscribers: Dict[Tuple[str, str], List[RollingQueue]] = defaultdict(list)
        
    def subscribe(self, inst: str, tf: str, max_len: int = 1024):
        queue = RollingQueue(max_len)
        self._subscribers[(inst, tf)].append(queue)
        return queue
    
    async def _process_payload(self, payload: dict):
        try:
            frame = FeatureFrame(**payload)
            
            queues = self._subscribers.get((frame.inst, frame.tf), None)
            if queues:
                for q in queues:
                    q.put_nowait(frame)
        except Exception as e:
            logger.error(f"DataRelay Error: {e} | Failed to process payload: {str(payload)[:50]}...")
    
    async def start(self, redis_subscriber):
        logger.info("Data Relay started...")
        await redis_subscriber.run_forever(self._process_payload)