# apps/run_manager_http.py
import asyncio, signal, os
import uvicorn
import contextlib

from utils.config import load_cfg
from feature.manager import WorkerManager
from feature.control import build_app


async def main():
    cfg = load_cfg()
    redis_dsn = "redis://:12345678@127.0.0.1:6379/0"
    stream = cfg.get("redis", {}).get("stream", "features")
    token = os.environ.get("CONTROL_TOKEN")

    manager = WorkerManager(cfg, redis_dsn=redis_dsn, stream_name=stream)
    await manager.start_from_cfg()
        
    app = build_app(manager, token=token)
    server = uvicorn.Server(
        uvicorn.Config(app, host=cfg.get("control", {}).get("host", "127.0.0.1"),
                            port=int(cfg.get("control", {}).get("port", 8080)),
                            loop="asyncio",
                            lifespan="off",
                            timeout_keep_alive=10,
                            log_config=None,
                            access_log=False)
    )

    http_task = asyncio.create_task(server.serve(), name="http")
    stop_event = asyncio.Event()

    def _graceful(*_):
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _graceful)
        except NotImplementedError:
            pass  # Windows

    await stop_event.wait()
    await manager.stop_all()
    http_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await http_task

if __name__ == "__main__":
    asyncio.run(main())
