# trading/app/trade_api.py
from trading.models import OrderRequest, Order
from trading.enums import Side, TdMode, OrdType, TimeInForce
from trading.idempotency import make_cl_ord_id, now_ms

class TradeAPI:
    """
    Application-facing trading API.
    Handles normalization, risk checks, idempotency/expTime, placement and optional waiting.
    """

    def __init__(self, 
                 instrument_svc, 
                 account_svc, 
                 exec_svc, 
                 risk_svc, 
                 reconcile_svc, 
                 event_bus, 
                 logger
                 ):
        self.instrument_svc = instrument_svc
        self.account_svc = account_svc
        self.exec_svc = exec_svc
        self.risk_svc = risk_svc
        self.reconcile_svc = reconcile_svc
        self.event_bus = event_bus
        self.log = logger

    async def place(self, req: OrderRequest, *, wait: bool = False, timeout_s: float = 3.0) -> Order:
        """
        Normalize → risk check → clOrdId/expTime → place → optional wait for reconcile.
        """
        # Normalize & validate
        if req.px is not None:
            req.px = self.instrument_svc.round_price(req.instId, req.px)
        req.sz = self.instrument_svc.normalize_size(req.instId, req.sz)
        self.instrument_svc.validate(req.instId, req.px, req.sz)

        # Risk checks
        await self.risk_svc.check_margins(req)
        # Optional: slippage via external best bid/ask injection from feature/datafeed

        # Idempotency & expiry
        req.clOrdId = req.clOrdId or make_cl_ord_id("algo")
        req.expTime = req.expTime or (now_ms() + 5000)

        # Place
        order = await self.exec_svc.place_order(req)

        # Wait for reconcile (ACK/final state)
        if wait:
            order = await self.reconcile_svc.reconcile_by_clOrdId(order.clOrdId, timeout_s=timeout_s)
        return order

    # ---- Convenience wrappers: NET mode (recommended) ----
    async def buy_open(self, instId: str, sz: float, *, px: float | None = None,
                       tdMode: TdMode = TdMode.CROSS,
                       ordType: OrdType = OrdType.LIMIT,
                       tif: TimeInForce | None = None,
                       wait: bool = False) -> Order:
        req = OrderRequest(instId=instId, side=Side.BUY, tdMode=tdMode, posSide="net",
                           ordType=ordType, sz=sz, px=px, tif=tif)
        return await self.place(req, wait=wait)

    async def sell_open(self, instId: str, sz: float, *, px: float | None = None,
                        tdMode: TdMode = TdMode.CROSS,
                        ordType: OrdType = OrdType.LIMIT,
                        tif: TimeInForce | None = None,
                        reduce_only: bool = False,
                        wait: bool = False) -> Order:
        req = OrderRequest(instId=instId, side=Side.SELL, tdMode=tdMode, posSide="net",
                           ordType=ordType, sz=sz, px=px, tif=tif, reduceOnly=reduce_only)
        return await self.place(req, wait=wait)

    # ---- Convenience wrappers: LONG/SHORT (dual position mode) ----
    async def open_long(self, instId: str, sz: float, *, px: float | None = None,
                        tdMode: TdMode = TdMode.CROSS,
                        ordType: OrdType = OrdType.LIMIT,
                        tif: TimeInForce | None = None,
                        wait: bool = False) -> Order:
        req = OrderRequest(instId=instId, side=Side.BUY, tdMode=tdMode, posSide="long",
                           ordType=ordType, sz=sz, px=px, tif=tif)
        return await self.place(req, wait=wait)

    async def open_short(self, instId: str, sz: float, *, px: float | None = None,
                         tdMode: TdMode = TdMode.CROSS,
                         ordType: OrdType = OrdType.LIMIT,
                         tif: TimeInForce | None = None,
                         wait: bool = False) -> Order:
        req = OrderRequest(instId=instId, side=Side.SELL, tdMode=tdMode, posSide="short",
                           ordType=ordType, sz=sz, px=px, tif=tif)
        return await self.place(req, wait=wait)

    async def close_long(self, instId: str, sz: float, *, px: float | None = None,
                         tdMode: TdMode = TdMode.CROSS,
                         ordType: OrdType = OrdType.LIMIT,
                         tif: TimeInForce | None = None,
                         wait: bool = False) -> Order:
        req = OrderRequest(instId=instId, side=Side.SELL, tdMode=tdMode, posSide="long",
                           ordType=ordType, sz=sz, px=px, tif=tif, reduceOnly=True)
        return await self.place(req, wait=wait)

    async def close_short(self, instId: str, sz: float, *, px: float | None = None,
                          tdMode: TdMode = TdMode.CROSS,
                          ordType: OrdType = OrdType.LIMIT,
                          tif: TimeInForce | None = None,
                          wait: bool = False) -> Order:
        req = OrderRequest(instId=instId, side=Side.BUY, tdMode=tdMode, posSide="short",
                           ordType=ordType, sz=sz, px=px, tif=tif, reduceOnly=True)
        return await self.place(req, wait=wait)

    # ---- Management helpers ----
    async def cancel_by_id(self, instId: str, ordId: str | None = None, clOrdId: str | None = None) -> Order:
        """Cancel order by ordId or clOrdId."""
        ...

    async def amend_price_size(self, instId: str, ordId: str | None,
                               new_px: float | None = None, new_sz: float | None = None) -> Order:
        """Amend order price/size."""
        ...

    async def get_open_orders(self, instId: str | None = None) -> list[Order]:
        """List open/pending orders."""
        ...

    async def get_positions(self) -> list:
        """Proxy to AccountService.get_positions()."""
        ...
