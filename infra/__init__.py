# infra/__init__.py
from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Protocol, Mapping, Any, Optional, Dict, AsyncGenerator, List

from infra.http_client import HttpClient

# ========== 1) 抽象端口：上层依赖这个，而非具体 HttpClient ==========
class HttpPort(Protocol):
    async def get_public(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]: ...
    async def get_private(self, path: str, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]: ...
    async def post_private(self, path: str, json_body: Mapping[str, Any]) -> Dict[str, Any]: ...
    async def delete_private(self, path: str, json_body: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]: ...
    async def iter_pages(self, path: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]: ...
    async def sync_server_time(self) -> int: ...


# ========== 2) 后台任务：周期对时 ==========
async def _periodic_time_sync(http: HttpClient, interval_sec: int = 600) -> None:
    while True:
        try:
            await http.sync_server_time()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            http.log.warning("Time sync failed: %s", e)
        await asyncio.sleep(interval_sec)


# ========== 3) 轻量“容器”：启动/维护/关闭 ==========
class HttpContainer:
    """
    负责 HttpClient 的创建、周期性对时任务、优雅关闭。
    - 组合根（应用入口）持有它。
    - 上层把 container.http 注入到各服务即可。
    """
    def __init__(self, http: HttpClient, tasks: List[asyncio.Task]) -> None:
        self.http = http
        self._tasks = tasks

    @classmethod
    async def start(cls, 
                    cfg: Mapping[str, Any],
                    logger: Optional[logging.Logger] = None,
                    api_key: Optional[str] = None,
                    secret_key: Optional[str] = None,
                    passphrase: Optional[str] = None,
                    *,
                    time_sync_interval_sec: int = 600
                    ) -> "HttpContainer":
        http = HttpClient(cfg, logger=logger, api_key=api_key, secret_key=secret_key, passphrase=passphrase)
        await http.sync_server_time()
        task_sync = asyncio.create_task(_periodic_time_sync(http, time_sync_interval_sec))
        return cls(http, tasks=[task_sync])
    
    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await t
        await self.http.close()

# ========== 4) 多客户端注册表（可选：同时接入 LIVE/PAPER/多交易所时用） ==========
class HttpClientRegistry:
    """
    用 name 维护多个 HttpClient 及其后台对时任务。
    典型用法：registry.add("okx-live", cfg_live, logger), registry.add("okx-paper", cfg_paper, logger)
    """
    def __init__(self) -> None:
        self._clients: Dict[str, HttpClient] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

    async def add(self, name: str,
                  cfg: Mapping[str, Any],
                  logger: Optional[logging.Logger] = None,
                  *,
                  time_sync_interval_sec: int = 600
                  ) -> HttpClient:
        if name in self._clients:
            return self._clients[name]
        cli = HttpClient(cfg, logger=logger)
        await cli.sync_server_time()
        task_sync = asyncio.create_task(_periodic_time_sync(cli, time_sync_interval_sec))
        self._clients[name] = cli
        self._tasks[name] = task_sync
        return cli
    
    def get(self, name: str) -> HttpClient:
        return self._clients[name]
    
    async def close_all(self) -> None:
        for t in self._tasks.values():
            t.cancel()
        for t in self._tasks.values():
            with contextlib.suppress(Exception):
                await t
        await asyncio.gather(*(c.close() for c in self._clients.values()), return_exceptions=True)
        self._clients.clear()
        self._tasks.clear()


# ========== 5) 简单健康检查（可用于启动探针/ready 探针） ==========
async def http_healthcheck(http: HttpPort) -> bool:
    try:
        await http.sync_server_time()
        return True
    except Exception:
        return False