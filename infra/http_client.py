# infra/http_client.py
from __future__ import annotations

import os
import aiohttp
import asyncio
import base64
import datetime as dt
import hashlib
import hmac
import json
import random
import time
from typing import Any, AsyncGenerator, Dict, Iterable, Mapping, Optional, Tuple
from urllib.parse import urlencode
import logging
from utils.logger import logger

JSON_SEPARATORS = (",", ":")

class HttpError(Exception):
    def __init__(self, status: int, message: str, payload: Optional[dict] = None):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.payload = payload or {}
    
        
class OkxApiError(Exception):
    def __init__(self, code: str, msg: str, payload: dict | None = None):
        self.code = code
        self.msg = msg
        self.payload = payload or {}
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        base = f"OKX API code={self.code}, msg={self.msg}"
        data = (self.payload.get("data") or [])
        if data:
            d0 = data[0] or {}
            s_code = d0.get("sCode")
            s_msg = d0.get("sMsg")
            if s_code or s_msg:
                base += f", sCode={s_code}, sMsg={s_msg}"
        return base


def _now_utc_iso_millis(ts_ms: Optional[int] = None) -> str:
    """
    OKX 要求 OK-ACCESS-TIMESTAMP 为 UTC ISO8601 毫秒格式: 2020-12-08T09:08:57.715Z
    参考：OKX v5 Overview 文档（签名/时间戳段落）. 
    """
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)
    # t = dt.datetime.utcfromtimestamp(ts_ms / 1000).replace(tzinfo=dt.timezone.utc)
    t = dt.datetime.fromtimestamp(ts_ms / 1000, tz=dt.timezone.utc)
    return t.isoformat(timespec="milliseconds").replace("+00:00", "Z")

def _json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, separators=JSON_SEPARATORS, ensure_ascii=False)

def _build_query(params: Optional[Mapping[str, Any]]) -> str:
    if not params:
        return ""
    return "?" + urlencode(params, doseq=True, safe=":/")

def _mask(s: Optional[str]) -> str:
    if not s:
        return ""
    if len(s) <= 8:
        return "*" * len(s)
    return s[:4] + "*" * (len(s) - 8) + s[-4:]

class HttpClient:
    def __init__(self,
                 cfg: Mapping[str, Any],
                 logger: Optional[logging.Logger] = None,
                 api_key: Optional[str] = None,
                 secret_key: Optional[str] = None,
                 passphrase: Optional[str] = None,
                 *,
                 timeout_ms: Optional[int] = None,
                 session: Optional[aiohttp.ClientSession] = None,
                 ) -> None:
        self.cfg = cfg
        self.log = logger or logging.getLogger("HttpClient")
        self.session = session
        self._owned_session = session is None

        oxk_cfg = cfg.get("okx", {})
        trading_cfg = cfg.get("trading", "PAPER")
        self.env = trading_cfg["mode"]

        self.base_url = oxk_cfg["rest_base"].get(self.env, "https://www.okx.com").rstrip("/")

        # credentials
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase =  passphrase

        # headers / paper
        self.demo_header_enabled: bool = bool(cfg.get("trading", {}).get("simulated_header", self.env == "paper"))
        # x-simulated-trading: 1 for Demo, 0 or absent for Live
        # 参见 OKX 文档/FAQ：Demo 交易需要 x-simulated-trading: 1 且使用 Demo APIKey。:contentReference[oaicite:0]{index=0}

        #meouts & retries
        timeouts_cfg = cfg.get("timeouts", {})
        retries_cfg = cfg.get("retries", {})
        self.timeout_ms = int(timeout_ms or timeouts_cfg.get("rest_ms", 3000))
        self.max_attempts = int(retries_cfg.get("rest_max_attempts", 3))
        self.backoff_ms = int(retries_cfg.get("backoff_ms", 200))

        self.clock_offset_ms: int = 0

        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout_ms / 1000)
            self.session = aiohttp.ClientSession(timeout=timeout, raise_for_status=False, trust_env=True)

        self.log.debug(
            f"HttpClient init base_url={self.base_url} env={self.env} key={_mask(self.api_key)}"
        )

    # ---- async context manager ----------------------------------------------------
    async def __aenter__(self) -> "HttpClient":
        if self._owned_session and (self.session is None or self.session.closed):
            timeout = aiohttp.ClientTimeout(total=self.timeout_ms / 1000.0)
            self.session = aiohttp.ClientSession(timeout=timeout, raise_for_status=False, trust_env=True)
        return self
    
    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owned_session and self.session is not None and not self.session.closed:
            await self.session.close()
    
    # ---- 时间同步 -----------------------------------------------------------------
    async def sync_server_time(self) -> int:
        """
        查询 /api/v5/public/time，校准时钟偏移（本地到服务器，毫秒）。
        官方建议：若本地时间与服务器有偏差，应先查询服务器时间再设置 timestamp。:contentReference[oaicite:1]{index=1}
        """
        path = "/api/v5/public/time"
        resp = await self.request("GET", path, auth=False, expect_okx=True)
        try:
            ts_server_ms = int(resp["data"][0]["ts"])
        except Exception as e:
            raise OkxApiError("time_parse_error", f"unexpected server time payload: {resp}") from e
        local_ms = int(time.time() * 1000)
        self.clock_offset_ms = ts_server_ms - local_ms
        self.log.info(f"Server time synced: offset_ms={self.clock_offset_ms}")
        return self.clock_offset_ms
    
    def _timestamp_iso(self) -> str:
        return _now_utc_iso_millis(int(time.time() * 1000) + self.clock_offset_ms)
    
    def _build_auth_headers(
            self, method: str, path: str, query: str, body: str
    ) -> Dict[str, str]:
        """
        OKX v5 签名串: timestamp + method + requestPath + body
        - requestPath 必须包含 path + querystring（若有）
        - body 为 JSON 串（GET 通常为空串）
        - 签名算法: HMAC-SHA256 -> Base64
        参考：OKX v5 文档与示例。:contentReference[oaicite:2]{index=2}
        """
        if not (self.api_key and self.secret_key and self.passphrase):
            raise OkxApiError("401", "missing API credentials")
        timestamp = self._timestamp_iso()
        request_path = path + (query or "")
        prehash = f"{timestamp}{method.upper()}{request_path}{body}"
        signature = base64.b64encode(
            hmac.new(
                self.secret_key.encode("utf-8"),
                prehash.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode()

        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": signature,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.demo_header_enabled and self.env == "paper":
            headers["x-simulated-trading"] = "1"

        return headers

    async def request(
            self,
            method: str,
            path: str,
            *,
            params: Optional[Mapping[str, Any]] = None,
            json_body: Optional[Mapping[str, Any]] = None,
            auth: bool = False,
            headers: Optional[Mapping[str, str]] = None,
            expect_okx: bool = True,
            timeout_ms: Optional[int] = None,
            retry: bool = True,
        ) -> Dict[str, Any]:
        """
        统一请求入口。
        - method: "GET" | "POST" | "DELETE" | "PUT"
        - path: 以 "/api/v5/..." 开头
        - params: querystring
        - json_body: JSON 请求体（会用于签名与实际发送）
        - auth: 是否使用 OKX 私有签名头
        - expect_okx: 是否期望 OKX 通用响应结构（包含 code/msg/data）
        - timeout_ms: 覆盖默认超时
        - retry: 遇到可重试错误时启用指数退避
        """
        assert path.startswith("/api/"), "path must start with /api/"
        method = method.upper()
        query = _build_query(params)
        url = self.base_url + path + query
        body_str = _json_dumps_compact(json_body) if json_body else ""
        req_headers = {
            "Content-Type": "application/json", "Accept": "application/json"
        }
        if headers:
            req_headers.update(headers)
        if auth:
            req_headers.update(self._build_auth_headers(method, path, query, body_str))

        timeout_ctx = aiohttp.ClientTimeout(total=timeout_ms / 1000.0 if timeout_ms else None)

        attempt = 0
        while True:
            attempt += 1
            try:
                async with self.session.request(
                    method,
                    url,
                    data=body_str if body_str else None,
                    headers=req_headers,
                    timeout=timeout_ctx,
                ) as resp:
                    text = await resp.text()
                    status = resp.status
                    if status >= 400:
                        if retry and (status >= 500 or status == 429) and attempt < self.max_attempts:
                            await self._sleep_backoff(attempt)
                            continue
                        raise HttpError(status, text)
                    
                    try:
                        payload = json.loads(text) if text else {}
                    except json.JSONDecodeError:
                        if expect_okx:
                            raise HttpError(status, f"invalid json: {text[:256]}")
                        return {"raw": text}
                    
                    if expect_okx:
                        code = str(payload.get("code", ""))
                        if code != "0":
                            if retry and attempt < self.max_attempts and code in {"50061", "50112", "51015"}:
                                await self._sleep_backoff(attempt)
                                continue
                            raise OkxApiError(code, payload.get("msg", ""), payload)
                        return payload
                    else:
                        return payload
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if retry and attempt < self.max_attempts:
                    await self._sleep_backoff(attempt)
                    logger.warning(f"Network error: {e} when requesting {url}, retrying...")
                    continue
                raise HttpError(599, f"Network error: {e}") from e
            except OkxApiError:
                raise
            except HttpError:
                raise
            except Exception as e:
                raise HttpError(599, f"Unexpected error: {e}") from e
            
    async def _sleep_backoff(self, attempt: int) -> None:
        base = self.backoff_ms * (2 ** (attempt - 1))
        jitter = random.randint(0, self.backoff_ms)
        await asyncio.sleep((base + jitter) / 1000.0)
    
    # ---- 便捷包装 -----------------------------------------------------------------
    async def get_public(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return await self.request("GET", path, params=params, auth=False, expect_okx=True)

    async def get_private(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return await self.request("GET", path, params=params, auth=True, expect_okx=True)

    async def post_private(self, path: str, json_body: Mapping[str, Any]) -> Dict[str, Any]:
        return await self.request("POST", path, json_body=json_body, auth=True, expect_okx=True)

    async def delete_private(self, path: str, json_body: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        return await self.request("DELETE", path, json_body=json_body, auth=True, expect_okx=True)
    
    # ---- Backfill / 分页通用迭代器 ------------------------------------------------
    async def iter_pages(
        self,
        path: str,
        *,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        auth: bool = False,
        cursor_param: str = "after",
        extract_next_cursor: Optional[callable] = None,
        limit_per_page: int = 100,
        max_pages: Optional[int] = None,
        begin_end: Optional[Tuple[int, int]] = None,  # (begin_ms, end_ms) 适配 begin/end 过滤
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        通用分页：
        - OKX 多数历史/列表接口使用 before/after 游标（ID/时间戳）；另一些支持 begin/end（毫秒）。
        - 默认每页 limit=100（OKX 常见最大值），支持设置。
        - extract_next_cursor: 回调，从 data 中提取下一页游标（若为空则中断）。
        - begin_end: 若提供，则自动在 params 中带上 begin/end（用于限定时间范围）。:contentReference[oaicite:4]{index=4}

        使用示例（行情 K线）：
            async for page in client.iter_pages(
                "/api/v5/market/candles",
                params={"instId": "ETH-USDT-SWAP", "bar": "1m"},
                auth=False,
                cursor_param="after",
                extract_next_cursor=lambda data: data[-1][0] if data else None,  # data[*][0] 通常为开/收/或 ts
            ):
                rows = page["data"]

        注意：不同接口对 cursor 字段的定义不同（ordId、ts 等），请据实际 payload 做提取逻辑。 
        """
        params = dict(params or {})
        params["limit"] = str(limit_per_page)
        if begin_end:
            params["begin"] = str(begin_end[0])
            params["end"] = str(begin_end[1])
        
        pages = 0
        next_cursor: Optional[str] = params.get(cursor_param)
        while True:
            if next_cursor:
                params[cursor_param] = str(next_cursor)
            
            payload = await self.request(
                method,
                path,
                params=params,
                json_body=json_body,
                auth=auth,
                expect_okx=True,
            )
            yield payload
            
            pages += 1
            if max_pages and pages >= max_pages:
                break
            
            data = payload.get("data") or []
            if not data:
                break

            if extract_next_cursor:
                next_cursor = extract_next_cursor(data)
            else:
                try:
                    next_cursor = data[-1][0]
                except Exception:
                    break

            if not next_cursor:
                break
    
    # ---- 常用封装 ---------------------------------------------------------
    # async def get_instruments(self, inst_type: str = "SWAP") -> Dict[str, Any]:
    #     """GET /public/instruments"""
    #     return await self.get_public("/api/v5/public/instruments", params={"instType": inst_type})

    # async def place_order(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
    #     """POST /trade/order（演示：保留到 ExecutionService 里更合理，这里仅做直通）"""
    #     return await self.post_private("/api/v5/trade/order", json_body=payload)

    # async def get_order(self, inst_id: str, cl_ord_id: str) -> Dict[str, Any]:
    #     """GET /trade/order（按 clOrdId 查询）"""
    #     return await self.get_private(
    #         "/api/v5/trade/order",
    #         params={"instId": inst_id, "clOrdId": cl_ord_id},
    #     )