# tests/test_instrument_service.py
import pytest
import asyncio
from decimal import Decimal
from trading.services.instrument_service import InstrumentService


class Endpoints:
    public_instruments = "/api/v5/public/instruments"

class FakeHttp:
    def __init__(self, payload):
        self._payload = payload

    async def get_public(self, path, params=None):
        assert path == Endpoints.public_instruments
        assert params and params.get("instType") == "SWAP"
        return self._payload

@pytest.mark.asyncio
async def test_refresh_and_get():
    payload = {
        "code": "0", "msg": "",
        "data": [
            {"instId": "BTC-USDT-SWAP", "tickSz": "0.1", "lotSz": "1", "minSz": "1", "ctVal": "1"},
            {"instId": "ETH-USDT-SWAP", "tickSz": "0.01", "lotSz": "0.1", "minSz": "0.1", "ctVal": "1"},
        ],
    }
    http = FakeHttp(payload)
    svc  = InstrumentService(http, Endpoints)

    await svc.refresh()
    btc = svc.get("BTC-USDT-SWAP")
    assert str(btc.tickSz) == "0.1"
    assert str(btc.lotSz)  == "1"

@pytest.mark.asyncio
async def test_round_and_validate_ok():
    payload = {
        "code":"0","msg":"",
        "data":[{"instId":"ETH-USDT-SWAP","tickSz":"0.01","lotSz":"0.1","minSz":"0.1","ctVal":"1"}],
    }
    svc = InstrumentService(FakeHttp(payload), Endpoints)
    await svc.refresh()

    px = svc.round_price("ETH-USDT-SWAP", 2034.0073)   # -> 2034.01
    sz = svc.normalize_size("ETH-USDT-SWAP", 0.2349)   # -> 0.2
    assert abs(px - 2034.01) < 1e-12
    assert abs(sz - 0.2)     < 1e-12

    # 不应抛异常
    svc.validate("ETH-USDT-SWAP", px=px, sz=sz)

@pytest.mark.asyncio
async def test_validate_violations():
    payload = {"code":"0","msg":"",
        "data":[{"instId":"XRP-USDT-SWAP","tickSz":"0.0005","lotSz":"1","minSz":"1","ctVal":"1"}]}
    svc = InstrumentService(FakeHttp(payload), Endpoints)
    await svc.refresh()

    # 数量小于 minSz
    with pytest.raises(Exception) as e1:
        svc.validate("XRP-USDT-SWAP", px=None, sz=0.9)
    assert "minSz" in str(e1.value)

    # 价格不对齐 tick，应给出修复建议
    with pytest.raises(Exception) as e2:
        svc.validate("XRP-USDT-SWAP", px=0.1234, sz=1)
    assert "tickSz" in str(e2.value)
