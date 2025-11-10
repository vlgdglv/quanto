# apps/run_manager_http.py
import asyncio, signal, os, argparse
import uvicorn
import contextlib

from utils.config import load_cfg
from feature.manager import WorkerManager
from feature.control import build_app

def env_default(name: str, default=None):
    return os.getenv(name, default)

def build_parser():
    p = argparse.ArgumentParser("feature-server")
    p.add_argument("--redis-dsn", default=env_default("REDIS_DSN", "redis://:12345678@127.0.0.1:6379/0"))
    p.add_argument("--stream",    default=env_default("REDIS_STREAM", None))
    p.add_argument("--host",      default=env_default("CONTROL_HOST", None))
    p.add_argument("--port",      type=int, default=int(env_default("CONTROL_PORT", "0") or 0))
    p.add_argument("--token",     default=env_default("CONTROL_TOKEN", None))
    p.add_argument("--config-path", default="configs/okx_feature_config.yaml")
    return p

async def main():
    args = build_parser().parse_args()
    
    cfg = load_cfg(args.config_path)
    
    redis_dsn = args.redis_dsn
    stream = args.stream or cfg.get("redis", {}).get("stream", None)

    host = args.host or cfg.get("control", {}).get("host", "127.0.0.1")
    port = args.port or int(cfg.get("control", {}).get("port", 24938))
    token = args.token

    manager = WorkerManager(cfg, redis_dsn=redis_dsn, stream_name=stream)
    await manager.start_from_cfg()
        
    app = build_app(manager, token=token)
    server = uvicorn.Server(
        uvicorn.Config(app, host=host,
                            port=port,
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
            pass

    await stop_event.wait()
    await manager.stop_all()
    http_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await http_task

if __name__ == "__main__":
    asyncio.run(main())
