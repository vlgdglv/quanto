# control/api.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from feature.manager import WorkerManager

def build_app(manager: WorkerManager, token: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="Feature Workers Control")

    def _auth(x_token: Optional[str]):
        if token and x_token != token:
            raise HTTPException(status_code=401, detail="unauthorized")

    class InstReq(BaseModel):
        inst: str

    class ChannelsPatch(BaseModel):
        inst: str
        # datafeed.channels 子树（例如 {"candles":{"bars":["1m","5m"],"fetch":true}}）
        channels: Dict[str, Any]

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/readyz")
    async def readyz():
        return {"ok": True}

    @app.get("/instruments")
    async def list_instruments(x_token: Optional[str] = Header(default=None)):
        _auth(x_token)
        return {"instruments": await manager.list_instruments()}

    @app.get("/status")
    async def get_status(x_token: Optional[str] = Header(default=None)):
        _auth(x_token)
        return await manager.status()

    @app.put("/inst")
    async def add_inst(req: InstReq, x_token: Optional[str] = Header(default=None)):
        _auth(x_token)
        await manager.add_instrument(req.inst)
        return {"ok": True}

    @app.delete("/inst")
    async def del_inst(req: InstReq, x_token: Optional[str] = Header(default=None)):
        _auth(x_token)
        await manager.remove_instrument(req.inst)
        return {"ok": True}

    @app.patch("/inst/channels")
    async def patch_inst_channels(req: ChannelsPatch, x_token: Optional[str] = Header(default=None)):
        _auth(x_token)
        frag = {"datafeed": {"channels": req.channels}}
        await manager.apply_delta_to(req.inst, frag)
        return {"ok": True}

    return app
