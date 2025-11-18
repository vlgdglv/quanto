#
import asyncio, os
from infra.ws_client  import WSClient
from trading.services.trading_service import OrdersFeed, OrderStore, TradingService, OrderCmd
from dotenv import load_dotenv

from infra import HttpContainer
from trading.services.instrument_service import InstrumentService
from trading.services.account_service import AccountService
from trading.services.endpoints import make_endpoints_from_cfg
from utils import logger, load_cfg

load_dotenv(os.path.expanduser("~/.okx/.env"))
OKX_API_KEY_PAPER = os.getenv("OKX_API_KEY_PAPER")
OKX_SECRET_PAPER = os.getenv("OKX_SECRET_KEY_PAPER")
OKX_PASSPHRASE_PAPER = os.getenv("OKX_PASSPHRASE_PAPER")

async def on_ev(ev: dict):
    print("[ORDERS EVENT]", ev)

async def test_place_and_cancel():
    cfg = load_cfg("configs/okx_account_config.yaml")

    mode = cfg["trading"]["mode"]
    assert mode == "paper", "Do fucking PAPER in test or you will be broked."
    
    endpoints = make_endpoints_from_cfg(cfg)
    container = await HttpContainer.start(cfg, logger, api_key=OKX_API_KEY_PAPER,
                                          secret_key=OKX_SECRET_PAPER,
                                          passphrase=OKX_PASSPHRASE_PAPER,
                                          time_sync_interval_sec=600)

    http = container.http
    inst_svc = InstrumentService(http, endpoints)
    acc_svc = AccountService(http, endpoints)
    await inst_svc.refresh("SWAP", "DOGE-USDT-SWAP")
    print(inst_svc._cache)
    
    trading_svc = TradingService(
        store=OrderStore(),
        inst_service=inst_svc,
        account_service=acc_svc,
        orders_feed=None,
        logger=logger,
    )
    
    cmd = OrderCmd(
        instId="DOGE-USDT-SWAP",
        clOrdId=trading_svc.gen_clOrdId(),
        side="buy",
        ordType="limit",
        tdMode="isolated",
        posSide="net",
        sz="1",
        px="0.15690",
        tag="mvp-test",
        leverage=10,
    )
    print(len(cmd.clOrdId))
    logger.info(f"submitting limit order: {cmd}")

    ack = await trading_svc.submit_limit(cmd, await_live=False)
    print("place ack:", ack)

    if not ack.accepted or not ack.ordId:
        logger.warning("place failed or no ordId returned, skip cancel")
        return

    # logger.info(f"canceling order instId={cmd.instId} ordId={ack.ordId}")
    # ok = await trading_svc.cancel(instId=cmd.instId, ordId=ack.ordId, clOrdId=cmd.clOrdId)
    # print("cancel result:", ok)

    await container.stop()
   
async def orderfeed():
    cfg = load_cfg("configs/okx_account_config.yaml")

    mode = cfg["trading"]["mode"]
    assert mode == "paper", "Do fucking PAPER in test or you will be broked."
    orders_args = [{
        "channel":"orders",
        "instType":"SWAP",   
    }]  
    orders_ws = WSClient(
        url=cfg["okx"]["ws"][mode]["private"],
        subscribe_args=orders_args,
        need_login=True,
        api_key=OKX_API_KEY_PAPER,
        secret_key=OKX_SECRET_PAPER,
        passphrase=OKX_PASSPHRASE_PAPER,
        inst_name="orders"
    )

    orders_feed = OrdersFeed(orders_ws)
    orders_feed.on_event(on_ev)

    task = await orders_feed.start()
    await task


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=str, default="orderfeed")
    
    args = parser.parse_args()
    if args.func == "orderfeed":
        asyncio.run(orderfeed())
    elif args.func == "test_place_and_cancel": 
        asyncio.run(test_place_and_cancel())
    else:
        raise NotImplementedError
