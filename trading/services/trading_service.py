# trading/services/execution_service.py
from typing import List, Optional
from trading.models import OrderRequest, Order, Fill
from trading.errors import OkxApiError

import logging

class OrderFeed:
    def __init__(self):
        self._ws = ws_client
        self._q: asyncio.Queue = asyncio.Queue(maxsize=8192)
        self._ws.bind_queue(self._q, put_timeout_ms=50, drop_when_full=True, microbatch=False)
        self._log = logger or logging.getLogger("OrdersFeed")
        self._cb: Optional[Callable[[Dict[str,Any]], Awaitable[None]]] = None
        self._task: Optional[asyncio.Task] = None
        
    def on_event(self, cb: Callable[[Dict[str,Any]], Awaitable[None]]):
        self._cb = cb

    async def start(self):
        if not self._cb:
            raise RuntimeError("OrdersFeed requires on_event callback before start()")
        self._task = asyncio.create_task(self._ws.run_forever())
        asyncio.create_task(self._drain())

    async def stop(self):
        if self._task:
            self._task.cancel()
        await self._ws.stop()

    async def _drain(self):
        while True:
            data = await self._q.get()
            if isinstance(data, list):
                for d in data:
                    if self._cb:
                        await self._cb(d)
            else:
                if self._cb:
                    await self._cb(data)


class OrderStore:
    def __init__(self):
        self._by_cl: Dict[str, Dict[str, Any]] = {}
        self._by_id: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def upsert_ws(self, ev: Dict[str, Any]):
        items = ev.get("data") if isinstance(ev, dict) else None
        if not items: 
            return
        async with self._lock:
            for it in items:
                cl = it.get("clOrdId")
                oid = it.get("ordId")
                st  = it.get("state")
                if cl:
                    cur = self._by_cl.get(cl, {})
                    cur.update(it)
                    self._by_cl[cl] = cur
                if oid:
                    cur = self._by_id.get(oid, {})
                    cur.update(it)
                    self._by_id[oid] = cur

    async def get_by_cl(self, clOrdId: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._by_cl.get(clOrdId)


class TradingService:
    def __init__(
        self,
        exec_svc,
        orders_feed: OrdersFeed,
        store: OrderStore,
        rules: InstrumentRules,
        logger: Optional[logging.Logger]=None,
        await_live_timeout: float = 3.0
    ):
        self._exec = exec_svc
        self._feed = orders_feed
        self._store = store
        self._rules = rules
        self._log = logger or logging.getLogger("TraderService")
        self._await_live_timeout = await_live_timeout
        self._wait_live: Dict[str, asyncio.Event] = {}

        self._feed.on_event(self._on_ws_event)

    async def start(self):
        await self._feed.start()

    async def stop(self):
        await self._feed.stop()

    async def submit_limit(self, cmd: OrderCmd, *, await_live: bool=True) -> OrderAck:
        if cmd.ordType not in ("limit","post_only","ioc","fok","optimal_limit_ioc"):
            raise ValueError("ordType must be a limit-like type")
        return await self._submit(cmd, await_live=await_live)

    async def submit_market(self, cmd: OrderCmd, *, await_live: bool=True) -> OrderAck:
        cmd.ordType = "market"
        cmd.px = None
        return await self._submit(cmd, await_live=await_live)

    async def cancel(self, instId: str, *, ordId: Optional[str]=None, clOrdId: Optional[str]=None) -> bool:
        try:
            await self._exec.cancel_order(instId, ordId=ordId, clOrdId=clOrdId)
            return True
        except Exception as e:
            self._log.warning("cancel failed: %s", e)
            return False

    async def _submit(self, cmd: OrderCmd, *, await_live: bool) -> OrderAck:
        if not cmd.clOrdId:
            cmd.clOrdId = self._gen_cl_id()

        r = await self._rules.get(cmd.instId)
        self._normalize(cmd, r)

        try:
            resp = await self._exec.place_order(cmd)
            ordId = None
            if isinstance(resp, dict):
                ordId = (resp.get("ordId") or (resp.get("data",[{}])[0].get("ordId")))
            else:
                ordId = getattr(resp, "ordId", None)
        except Exception as e:
            return OrderAck(instId=cmd.instId, clOrdId=cmd.clOrdId, ordId=None, accepted=False, msg=str(e))

        if await_live:
            ev = self._wait_live.setdefault(cmd.clOrdId, asyncio.Event())
            try:
                await asyncio.wait_for(ev.wait(), timeout=self._await_live_timeout)
            except asyncio.TimeoutError:
                self._log.warning("await_live timeout clOrdId=%s", cmd.clOrdId)

        return OrderAck(instId=cmd.instId, clOrdId=cmd.clOrdId, ordId=ordId, accepted=True)

    async def _on_ws_event(self, raw: Dict[str, Any]):
        if not isinstance(raw, dict):
            return
        ch = (raw.get("arg") or {}).get("channel")
        if ch != "orders":
            return
        
        await self._store.upsert_ws(raw)

        items = raw.get("data") or []
        for it in items:
            st = it.get("state")
            if st in ("live","partially_filled","filled"):
                cl = it.get("clOrdId")
                if cl and (cl in self._wait_live):
                    self._wait_live[cl].set()

    @staticmethod
    def _gen_cl_id() -> str:
        return f"TS-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"

    def _normalize(self, cmd: OrderCmd, r: Dict[str, Any]):
        def _step_round(x: float, step: float) -> float:
            return (int(x / step)) * step

        if cmd.ordType != "market":
            if not cmd.px:
                raise ValueError("limit-like order requires px")
            px = _step_round(float(cmd.px), float(r["px_tick"]))
            cmd.px = f"{px:.10f}".rstrip("0").rstrip(".")

        sz = _step_round(float(cmd.sz), float(r["sz_step"]))
        if sz < float(r["min_sz"]):
            raise ValueError(f"size<{r['min_sz']}")
        cmd.sz = f"{sz:.10f}".rstrip("0").rstrip(".")

        allowed = r.get("td_modes") or {"cross","isolated"}
        if cmd.tdMode not in allowed:
            raise ValueError(f"tdMode not allowed: {cmd.tdMode}")