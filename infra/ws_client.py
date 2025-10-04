# data/ws_public.py
from utils.logger import logger
import contextlib
import asyncio, json, time, websockets
from typing import Dict, Any, Callable, Awaitable, Iterable, Optional

Json = Dict[str, Any]

class WSClient:
    def __init__(self, 
        url: str,
        subscribe_args: Iterable[Json],
        need_login: bool = False,
        api_key: str = "",
        secret_key: str = "",
        passphrase: str = "",
        ping_interval: int = 20,
        reconnect_cap_s: int = 20,
    ):
        self.url = url
        self.args = list(subscribe_args)
        self.need_login = need_login
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.ping_interval = ping_interval
        self.reconnect_cap_s = reconnect_cap_s
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._stop = False

        logger.info(f"WSClient init url={url} need_login={need_login} "
                    f"ping_interval={ping_interval}s reconnect_cap_s={reconnect_cap_s} "
                    f"subs={len(list(subscribe_args))}")

    async def _login(self):
        """
        OKX WS 登录：sign = Base64(HMAC_SHA256(secret, timestamp + 'GET' + '/users/self/verify'))
        这里把签名细节外置/延后接入；若 need_login=False 则直接跳过
        """
        logger.info("WS login: start")

        import hmac, hashlib, base64
        ts = str(int(time.time()))

        msg = ts + 'GET' + '/users/self/verify'
        sign = base64.b64encode(hmac.new(self.secret_key.encode(), msg.encode(), hashlib.sha256).digest()).decode()
        payload = {
            "op": "login",
            "args":[{
                "apiKey": self.api_key,
                "passphrase": self.passphrase,
                "timestamp": ts,
                "sign": sign
            }]
        }
        await self._ws.send(json.dumps(payload))
        ack = await asyncio.wait_for(self._ws.recv(), timeout=5)
        data = json.loads(ack)
        if data.get("event") == "error":
            logger.error(f"WS login: failed ack={data}")
            raise RuntimeError(f"failed to login: {data}")
        logger.info("WS login: success")
        
    async def _subscribe(self):
        if not self.args: 
            logger.info("WS subscribe: no args, skip")
            return
        logger.info(f"WS subscribe: {self.args}")
        payload = {"op": "subscribe", "args": self.args}
        await self._ws.send(json.dumps(payload))
    
    async def _heartbeat(self):
        while not self._stop and self._ws:
            try:
                await self._ws.ping()
            except Exception:
                return
            await asyncio.sleep(self.ping_interval)
    
    async def run_forever(self, on_json: Callable[[Json], Awaitable[None]]):
        retry = 0

        while not self._stop:
            hb = None
            try:
                logger.info(f"WS connect: connecting to {self.url} (retry={retry})")
                async with websockets.connect(self.url, ping_interval=None, close_timeout=30) as ws:
                    self._ws = ws
                    logger.info("WS connect: connected")
                    if self.need_login:
                        await self._login()
                    await self._subscribe()
                    logger.info("WS subscribe: sent, start receiving")

                    hb = asyncio.create_task(self._heartbeat())
                    
                    # main read loop
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue

                        if "event" in data:
                            logger.info(f"WS {data.get('event')}: {data}")
                            continue

                        try:
                            await on_json(data)
                        except Exception:
                            logger.warning("WS queue full, dropping a message")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("WS loop: exception")
                backoff = min(self.reconnect_cap_s, 2 ** min(retry, 6))
                await asyncio.sleep(backoff)
                retry += 1
            finally:
                if hb:
                    hb.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await hb
                try:
                    if self._ws:
                        await self._ws.close()  
                except Exception:
                    pass
                finally:
                    self._ws = None
                logger.info("WS close: websocket closed")
                
    
    async def stop(self):
        self._stop = True   
        if self._ws:
            with contextlib.suppress(Exception):
                await self._ws.close()
            logger.info("WS stop: websocket closed")