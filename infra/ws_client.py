# data/ws_public.py
from utils.logger import logger
import contextlib
import asyncio, json, time, websockets
from typing import Dict, Any, Callable, Awaitable, Iterable, Optional, List

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
        inst_name: str = "",
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
        
        self._q: Optional[asyncio.Queue] = None
        self._put_timeout_ms = 50
        self._drop_when_full = True
        self._microbatch = False
        self._mb_maxn = 64
        self._mb_ms = 10

        self.inst_name = inst_name

        logger.info(f"WSClient {inst_name} init url={url} need_login={need_login} "
                    f"ping_interval={ping_interval}s reconnect_cap_s={reconnect_cap_s} "
                    f"subs={len(list(subscribe_args))}")
        
    def bind_queue(self, q: asyncio.Queue, *,
                   put_timeout_ms: int = 50,
                   drop_when_full: bool = True,
                   microbatch: bool = False,
                   microbatch_maxn: int = 64,
                   microbatch_ms: int = 10):
        self._q = q
        self._put_timeout_ms = put_timeout_ms
        self._drop_when_full = drop_when_full
        self._microbatch = microbatch
        self._mb_maxn = microbatch_maxn
        self._mb_ms = microbatch_ms

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
            logger.info(f"WS {self.inst_name} subscribe: no args, skip")
            return
        logger.info(f"WS subscribe request {self.args}")
        payload = {"op": "subscribe", "args": self.args}
        await self._ws.send(json.dumps(payload))

    async def _subscribe_args(self, args: List[dict]):
        if not args: return
        if not self._ws: return
        payload = {"op":"subscribe","args":args}
        logger.info(f"WS arg subscribe request {args}")
        await self._ws.send(json.dumps(payload))

    async def _unsubscribe_args(self, args: List[dict]):
        if not args: return
        if not self._ws: return
        payload = {"op":"unsubscribe","args":args}
        logger.info(f"WS arg unsubscribe request {args}")
        await self._ws.send(json.dumps(payload))

    async def _heartbeat(self):
        while not self._stop and self._ws:
            try:
                await self._ws.ping()
            except Exception:
                return
            await asyncio.sleep(self.ping_interval)
    
    async def run_forever(self, on_json: Callable[[Json], Awaitable[None]] = None):
        retry = 0
        while not self._stop:
            hb = None
            try:
                logger.info(f"WS {self.inst_name} connect: connecting to {self.url} (retry={retry})")
                async with websockets.connect(self.url, ping_interval=None, close_timeout=30) as ws:
                    self._ws = ws
                    logger.info(f"WS {self.inst_name} connect: connected")
                    if self.need_login:
                        await self._login()
                    await self._subscribe()
                    logger.info(f"WS {self.inst_name} subscribe: sent, start receiving")
                    hb = asyncio.create_task(self._heartbeat())
                    
                    if self._microbatch:
                        batch, last = [], asyncio.get_running_loop().time()
                        
                    # main read loop
                    async for msg in ws:
                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue

                        if "event" in data:
                            logger.info(f"WS event: {data.get('event')}: {data}")
                            continue
                        
                        if self._microbatch:
                            batch.append(data)
                            now = asyncio.get_running_loop().time()
                            if len(batch) >= self._mb_maxn or (now - last)*1000 >= self._mb_ms:
                                await self._q_put(batch); batch=[]; last=now
                        else:
                            await self._q_put(data)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"WS {self.inst_name} loop: exception")
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
                logger.info(f"WS {self.inst_name} close: websocket closed")
                
    async def _q_put(self, item):
        if not self._q: return
        try:
            if self._put_timeout_ms <= 0:
                self._q.put_nowait(item)
            else:
                await asyncio.wait_for(self._q.put(item), timeout=self._put_timeout_ms/1000)
        except (asyncio.TimeoutError, asyncio.QueueFull) as e:
            exc_type = type(e).__name__
            if self._drop_when_full:
                logger.warning(f"WS inst {self.inst_name} queue full/timeout ({exc_type}), drop 1 msg")
            else:
                await self._q.put(item) 
                
    async def stop(self):
        self._stop = True   
        if self._ws:
            with contextlib.suppress(Exception):
                await self._ws.close()
            logger.info("WS stop: websocket closed")