import os
import asyncio
import json
import httpx
from pprint import pprint

HOST = "127.0.0.1"
PORT = 8080
BASE = f"http://{HOST}:{PORT}"
TOKEN = None

INST = os.getenv("TEST_INST", "BTC-USDT-SWAP")

def _headers():
    h = {"content-type": "application/json"}
    if TOKEN:
        h["x-token"] = TOKEN
    return h

async def _req(client: httpx.AsyncClient, method: str, path: str, json_body=None):
    r = await client.request(method, f"{BASE}{path}", json=json_body, headers=_headers(), timeout=10.0)
    r.raise_for_status()
    if r.headers.get("content-type", "").startswith("application/json"):
        return r.json()
    return r.text

async def show_status(client: httpx.AsyncClient, title: str, inst: str=INST):
    st = await _req(client, "GET", "/status")
    print(f"\n=== {title} ===")
    inst_state = st.get(inst)
    if not inst_state:
        print(f"{inst} not running.")
        return
    summary = {
        "ws_clients": inst_state.get("ws_clients"),
        "queue_size": inst_state.get("queue_size"),
        "desired_counts": {k: v for k, v in (inst_state.get("desired") or {}).items()},
    }
    pprint(summary)

async def main():
    async with httpx.AsyncClient() as client:
        print("Health:", await _req(client, "GET", "/healthz"))
        print("Ready: ", await _req(client, "GET", "/readyz"))
        
        await show_status(client, "Just show it.")
        # 1) 初始 instruments
        before = await _req(client, "GET", "/instruments")
        print("\nBefore instruments:", before)

        # 2) 新增一个 instrument（如果已存在，服务端是幂等的）
        if False:
            print(f"\n[PUT /inst] add {INST}")
            await _req(client, "PUT", "/inst", {"inst": INST})

        # 等待 worker 启动并创建 WS 任务
        await asyncio.sleep(1.0)

        # 3) 设置/覆盖该 instrument 的 channels（含动态 bars）
        #    例子：开启 candles(kind=trade, bars=1m/5m/15m)、trades、books(level=5)，关闭其他
        if False:
            patch1 = {
                "inst": INST,
                "channels": {
                    "candles": {"fetch": True, "kind": "trade", "bars": ["1m", "15m", "15m", "1H"]},
                    "trades": {"fetch": True},
                    "books": {"fetch": True, "level": 5},
                    "funding-rate": {"fetch": True},
                    "open-interest": {"fetch": True},
                    "mark-price": {"fetch": False},
                    "index-tickers": {"fetch": False},
                    "liquidation-orders": {"fetch": False},
                    "price-limit": {"fetch": False},
                }
            }
            print(f"\n[PATCH /inst/channels] apply trade candles 1m/5m/15m + trades + books5")
            await _req(client, "PATCH", "/inst/channels", patch1)

            await asyncio.sleep(1.0)
            await show_status(client, "After PATCH #1")

        # 4) 动态减少频道 & 改 bars（只保留 1m，关闭 trades）
        if False:
            patch2 = {
                "inst": "XRP-USDT-SWAP",
                "channels": {
                    "candles": {"fetch": True, "kind": "trade", "bars": ["5m"]},
                    "trades": {"fetch": False},
                    "books": {"fetch": False, "level": 5}
                }
            }
            print(f"\n[PATCH /inst/channels] drop trades, keep books5, candles only 1m")
            await _req(client, "PATCH", "/inst/channels", patch2)

            await asyncio.sleep(1.0)
            await show_status(client, "After PATCH #2")

        # 5) 查看 instruments 列表
        print("\nInstruments:", await _req(client, "GET", "/instruments"))

        # 6) 删除 instrument
        if False:
            INST = "DOGE-USDT-SWAP"
            print(f"\n[DELETE /inst] remove {INST}")
            # 注意：httpx 允许 DELETE 带 json（FastAPI 端点定义用 Body）
            await _req(client, "DELETE", "/inst", {"inst": INST})

            await asyncio.sleep(0.5)
            await show_status(client, "After DELETE", INST)

        print("\nDone.")

if __name__ == "__main__":
    asyncio.run(main())
