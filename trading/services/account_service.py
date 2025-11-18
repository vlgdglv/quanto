# trading/services/account_service.py
import re, time
import uuid
import logging
from typing import Optional, Literal, Dict, Any, Callable, Awaitable, List

from trading.models import Position, Balance
from trading.enums import TdMode

StrNum = Optional[str]

def _to_float_or_none(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    x = str(x).strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None

def _to_float_zero_if_empty(x) -> float:
    v = _to_float_or_none(x)
    return v if v is not None else 0.0

class AccountService:
    """
    Account/position/leverage/mode configuration and queries.
    """
    def __init__(self, http_client, endpoints) -> None:
        self._http = http_client
        self._ep = endpoints
        self.log = getattr(http_client, "log", logging.getLogger("AccountService"))

    async def get_positions(self,  
                            instId: str,
                            instType:  Optional[str] = "SWAP",) -> List[Position]:
        path = getattr(self._ep, "account_positions", None) or \
               (self._ep.get("account_positions") if isinstance(self._ep, dict) else None) or \
               "/api/v5/account/positions"

        params: dict = {}
        inferred_type = self._inst_type_from_inst_id(instId) if instId else None

        instType_norm = instType.upper() if isinstance(instType, str) else None
        inferred_norm = inferred_type.upper() if isinstance(inferred_type, str) else None

        if instType_norm and inferred_norm and instType_norm != inferred_norm:
            self.log.warning(
                "get_positions: instType (%s) conflicts with inferred from instId (%s). "
                "Using instId-inferred type: %s",
                instType_norm, instId, inferred_norm
            )
            instType_norm = inferred_norm

        if instType_norm:
            params["instType"] = instType_norm
        if instId:
            params["instId"] = instId

        # Request
        resp = await self._http.get_private(path, params=params)
        raw_list = resp.get("data", []) or []

        positions: List[Position] = []
        for it in raw_list:
            if instType_norm and it.get("instType") != instType_norm:
                continue
            if instId and it.get("instId") != instId:
                continue

            positions.append(
                Position(
                    instType=it.get("instType", ""),
                    instId=it.get("instId", ""),
                    posId=it.get("posId", ""),
                    posSide=(it.get("posSide") or "net"),
                    mgnMode=(it.get("mgnMode") or ""),

                    pos=_to_float_zero_if_empty(it.get("pos")),
                    availPos=_to_float_zero_if_empty(it.get("availPos")),
                    avgPx=_to_float_or_none(it.get("avgPx")),
                    markPx=_to_float_or_none(it.get("markPx")),
                    liqPx=_to_float_or_none(it.get("liqPx")),
                    lever=_to_float_or_none(it.get("lever")),

                    upl=_to_float_or_none(it.get("upl")),
                    uplRatio=_to_float_or_none(it.get("uplRatio")),
                    notionalUsd=_to_float_or_none(it.get("notionalUsd")),

                    imr=_to_float_or_none(it.get("imr")),
                    mmr=_to_float_or_none(it.get("mmr")),
                    margin=_to_float_or_none(it.get("margin")),
                    mgnRatio=_to_float_or_none(it.get("mgnRatio")),

                    adl=int(it.get("adl")) if str(it.get("adl") or "").isdigit() else None,
                    cTime=int(it.get("cTime")) if str(it.get("cTime") or "").isdigit() else None,
                    uTime=int(it.get("uTime")) if str(it.get("uTime") or "").isdigit() else None,
                )
            )
        return positions

    async def get_balance(self, ccy: str = "USDT") -> Balance:
        """Return balance object for given currency."""
        path = getattr(self._ep, "account_balance", None) or \
            (self._ep.get("account_balance") if isinstance(self._ep, dict) else None) or \
            "/api/v5/account/balance"
        params: dict = {}
        if ccy:
            params["ccy"] = ccy
        resp = await self._http.get_private(path, params=params)
        balances = []
        data = resp.get("data", [])
        if not data:
            return balances
        details = (data[0] or {}).get("details", []) or []
        for d in details:
            ccy = d.get("ccy", "")
            equity = _to_float_or_none(d.get("eq"))
            avail = _to_float_or_none(d.get("availBal", None))
            if avail == 0.0:
                avail = _to_float_or_none(d.get("availEq", None))
            frozen = _to_float_or_none(d.get("frozenBal", None))
            ts = int(d.get("uTime", data[0].get("uTime", "0")) or 0)
            balances.append(Balance(ccy=ccy, equity=equity, avail=avail, frozen=frozen, ts=ts))
        return balances


    async def get_config(self) -> dict:
        """Return account config (position mode, Greeks, etc.)."""
        

    async def set_position_mode(self, net: bool) -> None:
        """Set position mode to net or long/short; requires no open pos/orders."""
        ...

    async def get_max_avail_size(self, instId: str, mgnMode: TdMode, ccy: Optional[str] = None) -> float:
        """Return maximum available size for order placement."""
        ...


    @staticmethod
    def _inst_type_from_inst_id(instId: str) -> Optional[str]:
        """
        根据 OKX instId 推断交易品种类型
        - BTC-USDT-SWAP       -> SWAP
        - BTC-USDT            -> SPOT
        - BTC-USD-230915      -> FUTURES
        - BTC-USD-230915-40000-C -> OPTION
        """
        if not instId or "-" not in instId:
            return None

        parts = instId.split("-")
        last = parts[-1].upper()

        if len(parts) == 2:
            return "SPOT"
        if last == "SWAP":
            return "SWAP"
        if re.fullmatch(r"\d{6}", last):
            return "FUTURES"
        if len(parts) >= 5 and last in ("C", "P"):
            return "OPTION"

        # other cases
        return None
    
    async def place_order(self,
        *,
        # Important and should be unique
        clOrdId: str,
        # Order params
        instId: str,
        side: Literal["buy","sell"],
        ordType: Literal["limit","market","post_only","ioc","fok","optimal_limit_ioc"],
        sz: StrNum,
        px: StrNum = None,
        tdMode: Literal["cross","isolated"],
        # relatively unimportant Order params
        posSide: Optional[Literal["net","long","short"]] = None,
        tag: Optional[str] = None,
        reduceOnly: Optional[bool] = None,
        tgtCcy: Optional[Literal["base_ccy","quote_ccy"]] = None,

        expTime: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        POST /api/v5/trade/order
        返回：{ "ordId": "...", "clOrdId": "...", "sCode": "0"/"xxx", "sMsg": "..." }
        仅代表“受理”；最终状态以订单WS为准。
        """
        path = getattr(self._ep, "trade_order", None) or \
               (self._ep.get("trade_order") if isinstance(self._ep, dict) else None) or \
               "/api/v5/trade/order"

        assert clOrdId is not None 

        payload: Dict[str, Any] = {
            "clOrdId": clOrdId,
            "instId": instId,
            "side": side,
            "ordType": ordType,
            "sz": str(sz),
            "tdMode": tdMode,
        }
                 
        if px is not None:            payload["px"] = str(px)
        if posSide is not None:       payload["posSide"] = posSide
        if reduceOnly is not None:    payload["reduceOnly"] = "true" if reduceOnly else "false"
        if tgtCcy is not None:        payload["tgtCcy"] = tgtCcy
        if tag is not None:           payload["tag"] = tag
        if expTime is not None:       payload["expTime"] = str(int(expTime))

        resp = await self._http.post_private(path, json_body=payload)
        
        code = str(resp.get("code", ""))
        if code != "0":
            raise RuntimeError(f"place_order http_okx code={code} msg={resp.get('msg')}")

        data = (resp.get("data") or [{}])[0]
        sCode = str(data.get("sCode", ""))
        if sCode != "0":
            sMsg = data.get("sMsg")
            return {
                "ordId": data.get("ordId"),
                "clOrdId": data.get("clOrdId") or clOrdId,
                "sCode": sCode,
                "sMsg": sMsg,
                "_accepted": False,
            }

        return {
            "ordId": data.get("ordId"),
            "clOrdId": data.get("clOrdId") or clOrdId,
            "sCode": sCode,
            "sMsg": data.get("sMsg"),
            "_accepted": True,
        }

    async def cancel_order(self,
        *,
        instId: str,
        ordId: Optional[str] = None,
        clOrdId: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        POST /api/v5/trade/cancel-order
        返回受理结果；最终是否取消以订单WS或查询为准。
        """
        path = getattr(self._ep, "trade_cancel", None) or \
               (self._ep.get("trade_cancel") if isinstance(self._ep, dict) else None) or \
               "/api/v5/trade/cancel-order"

        if not ordId and not clOrdId:
            raise ValueError("cancel_order requires ordId or clOrdId")

        payload = {"instId": instId}
        if ordId:   payload["ordId"] = ordId
        if clOrdId: payload["clOrdId"] = clOrdId

        resp = await self._http.post_private(path, json_body=payload)
        code = str(resp.get("code", ""))
        if code != "0":
            raise RuntimeError(f"cancel_order code={code} msg={resp.get('msg')}")
        return (resp.get("data") or [{}])[0]

    async def get_orders(self,
        *,
        instId: str,
        ordId: Optional[str] = None,
        clOrdId: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        GET /api/v5/trade/order
        """
        path = getattr(self._ep, "trade_get_order", None) or \
               (self._ep.get("trade_get_order") if isinstance(self._ep, dict) else None) or \
               "/api/v5/trade/order"
        params: Dict[str, Any] = {"instId": instId}
        if ordId:   params["ordId"] = ordId
        if clOrdId: params["clOrdId"] = clOrdId
        resp = await self._http.get_private(path, params=params)
        data = (resp.get("data") or [{}])[0]
        return data
    
    async def set_leverage(self, lever: int, mgnMode: str, 
                           instId: str, ccy: str=None, posSide: str=None):
        """
        GET /api/v5/account/set-leverage
        """
        path = getattr(self._ep, "account_set_leverage", None) or \
               (self._ep.get("account_set_leverage") if isinstance(self._ep, dict) else None) or \
               "/api/v5/account/set-leverage"
               
        if mgnMode not in ["isolated", "cross"]:
            raise ValueError("set_leverage mgnMode must be 'isolated' or 'cross'")
        
        params: Dict[str, Any] = {"lever": lever, "mgnMode": mgnMode}
        
        if instId: params["instId"] = instId
        if ccy: params["ccy"] = ccy
        if posSide: params["posSide"] = posSide
        resp = await self._http.post_private(path, json_body=params)
        data = (resp.get("data") or [{}])[0]
        return data
        
        
    async def list_open_orders(self, *, instId: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        GET /api/v5/trade/orders-pending
        冷启动/断线后刷新镜像。
        """
        path = getattr(self._ep, "trade_orders_pending", None) or \
               (self._ep.get("trade_orders_pending") if isinstance(self._ep, dict) else None) or \
               "/api/v5/trade/orders-pending"
        params: Dict[str, Any] = {}
        if instId: params["instId"] = instId
        resp = await self._http.get_private(path, params=params)
        return resp.get("data", []) or []

    async def list_fills(self, *, instId: str, after: Optional[int] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        GET /api/v5/trade/fills-history
        """
        path = getattr(self._ep, "trade_fills_history", None) or \
               (self._ep.get("trade_fills_history") if isinstance(self._ep, dict) else None) or \
               "/api/v5/trade/fills-history"
        params: Dict[str, Any] = {"instId": instId, "limit": str(int(limit))}
        if after is not None:
            params["after"] = str(int(after))
        resp = await self._http.get_private(path, params=params)
        return resp.get("data", []) or []
