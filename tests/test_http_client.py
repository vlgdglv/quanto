# tests/test_http_client.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import re
import time
import asyncio
import pytest
from aioresponses import aioresponses, CallbackResult

from infra.http_client import HttpClient, OkxApiError, HttpError

BASE = "https://www.okx.com"

def okx_payload(data=None, code="0", msg=""):
    return {"code": str(code), "msg": msg, "data": data if data is not None else []}

@pytest.mark.asyncio
async def test_sync_server_time(http_client: HttpClient):
    with aioresponses() as m:
        m.get(f"{BASE}/api/v5/public/time", payload=okx_payload([{"ts":"1700000000123"}]))
        offset = await http_client.sync_server_time()
        # offset = server - local，数值不断变化，这里只断言是整数即可
        assert isinstance(offset, int)

@pytest.mark.asyncio
async def test_public_get_instruments(http_client: HttpClient):
    with aioresponses() as m:
        m.get(
            f"{BASE}/api/v5/public/instruments?instType=SWAP",
            payload=okx_payload([{"instId":"ETH-USDT-SWAP","tickSz":"0.1","lotSz":"1","minSz":"1"}])
        )
        resp = await http_client.get_instruments("SWAP")
        assert resp["code"] == "0"
        assert resp["data"][0]["instId"] == "ETH-USDT-SWAP"

@pytest.mark.asyncio
async def test_private_get_with_signature_and_paper_header(http_client: HttpClient, monkeypatch):
    """
    验证私有请求：必须带 OK-ACCESS-* 头，且 PAPER 环境包含 x-simulated-trading: 1。
    通过 monkeypatch 固定 timestamp，拦截请求以断言 headers。
    """
    fixed_ts = "2024-01-01T00:00:00.000Z"
    monkeypatch.setattr(http_client, "_timestamp_iso", lambda: fixed_ts)

    def _assert_request(url, **kwargs):
        headers = kwargs["headers"]
        # assert headers["OK-ACCESS-KEY"] == "test_api_key"
        # assert headers["OK-ACCESS-PASSPHRASE"] == "test_pass"
        assert headers["OK-ACCESS-SIGN"]          # 有值即可，签名正确性在下方单测
        assert headers["OK-ACCESS-TIMESTAMP"] == fixed_ts
        assert headers.get("x-simulated-trading") == "1"
        return CallbackResult(
            status=200,
            payload=okx_payload([{"dummy": "ok"}]),
            headers={"Content-Type": "application/json"},
        )

    with aioresponses() as m:
        m.get(f"{BASE}/api/v5/account/balance", callback=_assert_request)
        resp = await http_client.get_private("/api/v5/account/balance")
        assert resp["code"] == "0"
        assert resp["data"][0]["dummy"] == "ok"

@pytest.mark.asyncio
async def test_signature_exact_value_for_post(http_client: HttpClient, monkeypatch):
    """
    计算一个可复现的签名值，严格断言 SIGN 的 Base64。
    公式：HMAC_SHA256(timestamp + method + requestPath + body), key=secret
    """
    # 固定 timestamp
    fixed_ts = "2024-01-01T00:00:00.000Z"
    monkeypatch.setattr(http_client, "_timestamp_iso", lambda: fixed_ts)

    payload = {
        "instId":"ETH-USDT-SWAP",
        "tdMode":"cross",
        "clOrdId":"unit-test-0001",
        "side":"buy",
        "ordType":"limit",
        "px":"2500",
        "sz":"1"
    }
    # 预计算签名（与 HttpClient 内部逻辑一致）
    import base64, hmac, hashlib
    prehash = fixed_ts + "POST" + "/api/v5/trade/order" + json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    expected = base64.b64encode(
        hmac.new(http_client.secret_key.encode(), prehash.encode(), hashlib.sha256).digest()
    ).decode()

    def _assert_post(url, **kwargs):
        headers = kwargs["headers"]
        assert headers["OK-ACCESS-SIGN"] == expected
        return CallbackResult(
            status=200,
            payload=okx_payload([{"ordId":"123"}]),
            headers={"Content-Type": "application/json"},
        )

    with aioresponses() as m:
        m.post(f"{BASE}/api/v5/trade/order", callback=_assert_post)
        resp = await http_client.place_order(payload)  # 直通方法
        assert resp["code"] == "0"
        assert resp["data"][0]["ordId"] == "123"

@pytest.mark.asyncio
async def test_retry_on_429_and_5xx(http_client: HttpClient, monkeypatch):
    """
    首两次返回 429/500，第三次成功。验证指数退避重试路径。
    为了加快测试，将 backoff sleep 替换为 no-op。
    """
    monkeypatch.setattr(http_client, "_sleep_backoff", lambda attempt: asyncio.sleep(0))

    calls = {"n": 0}
    def _flaky(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return CallbackResult(
                status=429,
                headers={"Content-Type": "application/json"},
                payload={"code": "429", "msg": "rate limit"},
            )
        if calls["n"] == 2:
            return CallbackResult(
                status=500,
                headers={"Content-Type": "application/json"},
                payload={"code": "500", "msg": "server error"},
            )
        return CallbackResult(
            status=200,
            payload=okx_payload([{"ok": True}]),
            headers={"Content-Type": "application/json"},
        )

    with aioresponses() as m:
        m.get("https://www.okx.com/api/v5/account/positions?instId=ETH-USDT-SWAP", callback=_flaky, repeat=True)
        resp = await http_client.get_private("/api/v5/account/positions", params={"instId":"ETH-USDT-SWAP"})
        assert resp["code"] == "0"
        assert resp["data"][0]["ok"] is True
        assert calls["n"] == 3

@pytest.mark.asyncio
async def test_business_error_no_retry(http_client: HttpClient):
    """
    业务错误（参数无效等 code != 0 且不在重试名单）直接抛 OkxApiError。
    """
    with aioresponses() as m:
        m.get(f"{BASE}/api/v5/trade/order?instId=ETH-USDT-SWAP&clOrdId=bad",
              payload={"code":"51001","msg":"Parameter error","data":[]})
        with pytest.raises(OkxApiError) as ei:
            await http_client.get_private("/api/v5/trade/order", params={"instId":"ETH-USDT-SWAP","clOrdId":"bad"})
        assert ei.value.code == "51001"

@pytest.mark.asyncio
async def test_http_error_status_raises(http_client: HttpClient):
    with aioresponses() as m:
        m.get(f"{BASE}/api/v5/public/instruments?instType=SWAP", status=503, body="svc unavailable")
        with pytest.raises(HttpError) as ei:
            await http_client.get_public("/api/v5/public/instruments", params={"instType":"SWAP"})
        assert ei.value.status >= 500

@pytest.mark.asyncio
async def test_iter_pages_default_cursor(http_client: HttpClient):
    """
    验证通用分页：默认以 data[-1][0] 作为下一页游标，最多 3 页后停止。
    """
    pages = [
        okx_payload([["c1","other"],["c2","other"]]),
        okx_payload([["c3","other"],["c4","other"]]),
        okx_payload([["c5","other"]]),
    ]
    with aioresponses() as m:
        m.get(re.compile(f"{BASE}/api/v5/market/candles.*"), payload=pages[0])
        m.get(re.compile(f"{BASE}/api/v5/market/candles.*after=c2.*"), payload=pages[1])
        m.get(re.compile(f"{BASE}/api/v5/market/candles.*after=c4.*"), payload=pages[2])

        seen = []
        async for payload in http_client.iter_pages(
            "/api/v5/market/candles",
            params={"instId":"ETH-USDT-SWAP","bar":"1m"},
            auth=False,
            cursor_param="after",
            max_pages=3,
        ):
            seen.append(payload["data"])

        assert len(seen) == 3
        assert seen[0][0][0] == "c1"
        assert seen[-1][-1][0] == "c5"

@pytest.mark.asyncio
async def test_iter_pages_with_begin_end_and_custom_extractor(http_client: HttpClient):
    """
    验证 begin/end 参数注入与自定义游标提取函数。
    """
    # 两页数据，每页最后一条的第 0 项作为下一页游标
    page1 = okx_payload([[1700000000000, "row1"], [1700000001000, "row2"]])
    page2 = okx_payload([[1700000002000, "row3"]])

    with aioresponses() as m:
        m.get(re.compile(f"{BASE}/api/v5/market/candles.*begin=1699999999000&end=1700000003000.*"),
              payload=page1)
        m.get(re.compile(f"{BASE}/api/v5/market/candles.*after=1700000001000.*"),
              payload=page2)

        rows = []
        async for payload in http_client.iter_pages(
            "/api/v5/market/candles",
            params={"instId":"ETH-USDT-SWAP","bar":"1m"},
            auth=False,
            cursor_param="after",
            extract_next_cursor=lambda data: data[-1][0] if data else None,
            begin_end=(1699999999000, 1700000003000),
            max_pages=2,
        ):
            rows.extend(payload["data"])

        assert len(rows) == 3
        assert rows[0][1] == "row1"
        assert rows[-1][1] == "row3"
