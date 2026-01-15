# trading/services/execution_service.py
from typing import List, Optional, Dict, Any, Callable, Awaitable
import asyncio
import time, uuid
from trading.models import OrderCmd, OrderAck, Instrument
from trading.services.account_service import AccountService
from trading.services.instrument_service import InstrumentService
from trading.errors import OkxApiError
from infra.ws_client import WSClient
import logging


class OrdersFeed:
    def __init__(self, ws_client: WSClient, logger: Optional[logging.Logger]=None):
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
        task = asyncio.create_task(self._drain())
        return task

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
        store: OrderStore,
        inst_service: InstrumentService,
        account_service: AccountService,
        orders_feed: OrdersFeed=None,
        logger: Optional[logging.Logger]=None,
        await_live_timeout: float = 3.0
    ):
        self.account_service = account_service
        self.inst_service = inst_service

        self._store = store
        self._log = logger or logging.getLogger("TraderService")
        self._await_live_timeout = await_live_timeout
        self._wait_live: Dict[str, asyncio.Event] = {}

        
        self._feed = orders_feed

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
            await self.account_service.cancel_order(instId=instId, ordId=ordId, clOrdId=clOrdId)
            return True
        except Exception as e:
            self._log.warning(f"cancel failed: {e}")
            return False

    async def _submit(self, cmd: OrderCmd, *, await_live: bool) -> OrderAck:
        if not cmd.clOrdId:
            cmd.clOrdId = self.gen_clOrdId()

        await self.inst_service.get_or_refresh(cmd.instId)
        cmd = self._normalize(cmd)

        if cmd.leverage is not None:
            resp = await self.account_service.set_leverage(instId=cmd.instId, lever=cmd.leverage, mgnMode=cmd.tdMode)
            print(resp)
            
        try:
            # Unpack cmd hear
            resp = await self.account_service.place_order(
                clOrdId=cmd.clOrdId,
                instId=cmd.instId,
                side=cmd.side,
                ordType=cmd.ordType,
                sz=cmd.sz,
                px=cmd.px,
                tdMode=cmd.tdMode,
            )
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
    def gen_clOrdId() -> str:        
        ms = str(int(time.time() * 1000)) 
        u = uuid.uuid4().hex
        result = ms + u[:32 - len(ms)] 
        return result

    def _normalize(self, cmd: OrderCmd):
        inst_id = cmd.instId

        px_f: float | None = None
        if cmd.ordType != "market":
            if cmd.px is None:
                raise ValueError("limit-like order requires px")

            px_f = self.inst_service.round_price(inst_id, float(cmd.px))
            cmd.px = f"{px_f:.10f}".rstrip("0").rstrip(".")
            
        sz_f = self.inst_service.normalize_size(inst_id, float(cmd.sz))
        cmd.sz = f"{sz_f:.10f}".rstrip("0").rstrip(".")

            
        self.inst_service.validate(inst_id, px_f, sz_f)

        return cmd

        # inst = self.inst_service.get(inst_id)

        # allowed = getattr(inst, "td_modes", {"cross", "isolated"})
        # if cmd.tdMode not in allowed:
        #     raise ValueError(f"tdMode not allowed for {inst_id}: {cmd.tdMode}")