# 
import asyncio, contextlib, json
from typing import Dict, Any, List, Tuple
import datetime, random, hashlib
from zoneinfo import ZoneInfo

from infra.ws_client import WSClient
from infra.redis_stream import RedisStreamsPublisher
from feature.engine_pd import FeatureEnginePD
from feature.processor import FeatureEngineProcessor
from feature.provider import build_args_for_one_inst_grouped_by_kind
from utils.logger import logger


class InstrumentWorker:
    """
    一个 inst 至多维护 2 条 WS（public/business），共享一个内部队列与一个 Processor。
    支持：启动时从 cfg 构建订阅；运行时 apply_delta 动态增删（频道/TF）。
    """
    _tz = ZoneInfo("Asia/Shanghai")

    def __init__(self, 
                 inst: str, 
                 cfg: Dict[str, Any], 
                 redis_dsn: str, 
                 stream_name: str = None
                 ):
        self.inst = inst
        self.cfg = cfg
        self._q: asyncio.Queue = asyncio.Queue(maxsize=cfg.get("runtime", {}).get("queue_max", 8192 * 8))
        
        stream_name = stream_name if stream_name else inst

        engine = FeatureEnginePD(enable_summary=True)
        pub = RedisStreamsPublisher(dsn=redis_dsn, stream=stream_name)
        self.processor = FeatureEngineProcessor(cfg, engine, publisher=pub, stream_name=stream_name)

        # ws_kind -> WSClient
        self.clients: Dict[str, WSClient] = {}
        self._tasks: List[asyncio.Task] = []

        # ws_kind -> set of arg tuple
        self._desired: Dict[str, set] = {"public": set(), "business": set()}
    
        

    def _url_for_kind(self, ws_kind: str) -> str:
        from feature.provider import build_ws_url
        return build_ws_url(self.cfg, ws_kind)

    def _auth_bundle(self) -> Dict[str, str]:
        df_auth = self.cfg.get("datafeed", {})
        auth = self.cfg.get("auth", {})
        return {
            "api_key": df_auth.get("api_key", auth.get("api_key", "")),
            "secret_key": df_auth.get("secret_key", auth.get("secret_key", "")),
            "passphrase": df_auth.get("passphrase", auth.get("passphrase", "")),
            "need_login": bool(df_auth.get("need_login", False)),
        }

    async def start(self):
        initial = build_args_for_one_inst_grouped_by_kind(self.cfg, self.inst)  # ws_kind -> args[]
        auth = self._auth_bundle()

        for ws_kind, args in initial.items():
            if not args: continue
            url = self._url_for_kind(ws_kind)
            c = WSClient(
                url=url,
                subscribe_args=args,
                need_login=auth["need_login"],
                api_key=auth["api_key"], secret_key=auth["secret_key"], passphrase=auth["passphrase"],
                ping_interval=15,
                inst_name=f"{self.inst}_{ws_kind}",
            )
            c.bind_queue(self._q, put_timeout_ms=50, drop_when_full=True, microbatch=False, microbatch_maxn=64, microbatch_ms=10)
            self.clients[ws_kind] = c
            self._desired[ws_kind] = { (a["channel"], a["instId"]) for a in args }
            self._tasks.append(asyncio.create_task(c.run_forever()))
        
        self._tasks.append(asyncio.create_task(
            self._scheduled_reconnect(
                hours=set([0, 4, 8, 12, 15, 16, 20]),
                minutes_before=5,
                jitter_each=0.6,
            )
        ))
        self._tasks.append(asyncio.create_task(self._consume_loop()))

    async def stop(self):
        for t in self._tasks:
            t.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _consume_loop(self):
        while True:
            item = await self._q.get()
            try:
                if isinstance(item, list):
                    for m in item: await self.processor.handle(m)
                else:
                    await self.processor.handle(item)
            finally:
                self._q.task_done()

    async def apply_delta(self, new_cfg_fragment: Dict[str, Any]):
        """
        传入的 new_cfg_fragment 仅包含 datafeed.channels 的局部变化（例如 add/remove 某个 bar 或切换 candles.kind）。
        也可以传入完整 cfg（按 inst 重建 args，再做 diff）。
        """
        # 1) 合成一个“更新后”的临时 cfg
        merged = json.loads(json.dumps(self.cfg))
        # print("In inst worker new_cfg_fragment: ", new_cfg_fragment)
        if "datafeed" in new_cfg_fragment:
            merged["datafeed"] = {**self.cfg.get("datafeed", {}), **new_cfg_fragment["datafeed"]}

        # print("In inst worker merged: ", merged)
        # 2) 重新计算该 inst 的目标 args
        target = build_args_for_one_inst_grouped_by_kind(merged, self.inst)  # ws_kind -> args[]
        # print("In inst worker target: ", target)
        
        all_ws_kind = set(target.keys()) | set(self._desired.keys()) 
        # 3) 对每个 ws_kind 做增删差量
        for ws_kind in all_ws_kind:
            args = target.get(ws_kind, set())
            want = { (a["channel"], a["instId"]) for a in args }
            have = self._desired.get(ws_kind, set())
            add = want - have
            rem = have - want

            if (add or rem) and ws_kind not in self.clients:
                # ws_kind 原先没有连接（例如从未订阅过 business），先起连接
                url = self._url_for_kind(ws_kind)
                auth = self._auth_bundle()
                c = WSClient(url=url, subscribe_args=[],
                             need_login=(ws_kind=="business") or auth["need_login"],
                             api_key=auth["api_key"], secret_key=auth["secret_key"], passphrase=auth["passphrase"],
                             ping_interval=15)
                c.bind_queue(self._q, put_timeout_ms=50, drop_when_full=True, microbatch=True)
                self.clients[ws_kind] = c
                self._tasks.append(asyncio.create_task(c.run_forever()))
                self._desired[ws_kind] = set()

            client = self.clients.get(ws_kind)
            if not client:
                continue

            # 4) 逐类发包（先 add 再 remove，降低短暂漏订风险）
            if add:
                add_args = [ {"channel": ch, "instId": iid} for (ch, iid) in add ]
                await client._subscribe_args(add_args)
            if rem:
                rem_args = [ {"channel": ch, "instId": iid} for (ch, iid) in rem ]
                await client._unsubscribe_args(rem_args)

            # 5) 刷新本地 desired
            self._desired[ws_kind] = want

        # 6) 更新保存 cfg
        self.cfg = merged
    
    async def _scheduled_reconnect(
        self, *, hours: set[int], minutes_before: int = 5, jitter_each: float = 0.4
    ):
        assert all(0 <= h <= 23 for h in hours)
        
        inst_spread = int.from_bytes(hashlib.md5(str(self.inst).encode('utf-8')).digest()[:4], 'big') % max(1, 30)
        while True:
            now = datetime.datetime.now(tz=self._tz)
            candidates = []
            for d in (0, 1):
                day = (now + datetime.timedelta(days=d)).date()
                for h in sorted(hours):
                    tgt = datetime.datetime(day.year, day.month, day.day, h, 0, 0, tzinfo=self._tz)
                    fire = tgt - datetime.timedelta(minutes=minutes_before, seconds=inst_spread)
                    if fire > now:
                        candidates.append(fire)
            next_fire = min(candidates)
            await asyncio.sleep((next_fire - now).total_seconds())

            await self._reconnect_all(
                reason=f"pre_{next_fire.astimezone(self._tz).strftime('%H%M')}",
                jitter_each=jitter_each
            )

            await asyncio.sleep(inst_spread + 2)

    async def _reconnect_all(self, *, reason: str = "scheduled", jitter_each: float = 0.4):
        for i, (ws_kind, client) in enumerate(self.clients.items()):
            await asyncio.sleep(random.uniform(0, jitter_each) + i * 0.05)
            try:
                await client.force_reconnect(reason=reason)
            except Exception:
                pass