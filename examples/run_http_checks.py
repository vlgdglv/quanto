# examples/run_instrument_smoke.py
import logging, yaml, random
from pathlib import Path
import asyncio, aiohttp

from infra import HttpContainer
from trading.services.instrument_service import InstrumentService
from trading.services.account_service import AccountService
from trading.services.endpoints import make_endpoints_from_cfg
from utils import logger, load_cfg

import os
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.okx/.env"))
OKX_API_KEY = os.getenv("OKX_API_KEY")
OKX_SECRET = os.getenv("OKX_SECRET_KEY")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE")

OKX_API_KEY_PAPER = os.getenv("OKX_API_KEY_PAPER")
OKX_SECRET_PAPER = os.getenv("OKX_SECRET_KEY_PAPER")
OKX_PASSPHRASE_PAPER = os.getenv("OKX_PASSPHRASE_PAPER")


async def test_instruments():
    cfg = load_cfg("configs/okx_trading_config.yaml")
    endpoints = make_endpoints_from_cfg(cfg)
    container = await HttpContainer.start(cfg, logger, time_sync_interval_sec=600)

    http = container.http
    svc = InstrumentService(http, endpoints)
    # Test SPOT
    try:
        if True:
            shit = await svc.get_inst_price("DOGE-USDT-SWAP")
            print(shit)
        if False:
            await svc.refresh("SPOT")
            print("✅ instruments cached, total caches: %d" % len(svc._cache))
        if False:
            await svc.refresh("SWAP", "BTC-USDT-SWAP")
            await svc.refresh("SPOT", "DOGE-USDT")
            await svc.refresh("SPOT", "BTC-USDT")
            print("✅ instruments cached, caches: ", svc._cache)
        if False:

            # 1) 拉取并建缓存
            await svc.refresh("SWAP")
            print("✅ instruments cached")

            # 2) 取若干样本做冒烟校验
            #    - 优先 USDT 合约，数量不大于 5 个
            wanted = []
            for instId, _ in list(svc._cache.items()):
                if instId.endswith("-SWAP") and ("USDT" in instId or "USDC" in instId):
                    wanted.append(instId)
                if len(wanted) >= 5:
                    break
            if not wanted:
                # 兜底：随机抽 3 个
                wanted = [k for k in list(svc._cache.keys())[:3]]

            print(f"🧪 sample instruments: {wanted}")

            failures = 0
            for instId in wanted:
                inst = svc.get(instId)
                # 价格按 tick 四舍五入
                px0 =  random.uniform(1, 5000)
                px   =  svc.round_price(instId, px0)
                # 数量按 lot 向下规整
                sz0 =  random.uniform(float(inst.lotSz), float(inst.lotSz) * 50)
                sz   =  svc.normalize_size(instId, sz0)

                # validate 不应抛异常（把 px 传入以校验 price 精度）
                try:
                    svc.validate(instId, px=px, sz=sz)
                except Exception as e:
                    failures += 1
                    print(f"❌ {instId} failed: {e}")
                    continue

                print(f"✅ {instId}: px0={px0:.6f} -> px={px:.6f} | sz0={sz0:.8f} -> sz={sz:.8f} "
                    f"(tickSz={inst.tickSz}, lotSz={inst.lotSz}, minSz={inst.minSz})")

            if failures:
                raise SystemExit(f"SMOKE FAILED: {failures} instruments failed")
            print("🎉 SMOKE OK")
    finally:
        await container.stop()


async def test_accounts():
    cfg = load_cfg("configs/okx_account_config.yaml")
    endpoints = make_endpoints_from_cfg(cfg)
    container = await HttpContainer.start(cfg, logger, 
                                          api_key=OKX_API_KEY_PAPER,
                                          secret_key=OKX_SECRET_PAPER,
                                          passphrase=OKX_PASSPHRASE_PAPER,
                                          time_sync_interval_sec=600)

    http = container.http
    svc = AccountService(http, endpoints)
    # Test SPOT
    try:
        print("Test get_positions()")
        position = await svc.get_positions(instId="DOGE-USDT-SWAP")
        print(position)
        
        print("Test get_balance()")
        balance = await svc.get_balance(ccy="USDT")
        print(balance)
        # print("Test get_orders()")
        # order = await svc.get_orders(instId="DOGE-USDT-SWAP")
        # print(order)
    finally:
        await container.stop()

async def ping_okx():
    timeout = aiohttp.ClientTimeout(total=5)  # 关键：总超时
    async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as s:
        async with s.get("https://www.okx.com/api/v5/public/time") as r:
            print(r.status, await r.text())

if __name__ == "__main__":
    # asyncio.run(test_instruments())
    asyncio.run(test_accounts())
