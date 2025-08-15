# data/normalizer.py
from datetime import datetime, timezone
from typing import Dict, Any, Callable, List, Tuple
import pandas as pd

HandlerFunc = Callable[[Dict[str, Any]], pd.DataFrame]

channel_registry: Dict[str, HandlerFunc] = {}

def register_channel(prefix: str):
    def decorator(fn: HandlerFunc):
        channel_registry[prefix] = fn
        return fn
    return decorator

def _dt(ms: int):
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc)

@register_channel("candle")
@register_channel("mark-price-candle")
@register_channel("index-candle")
def handle_candle(msg: Dict[str, Any]) -> pd.DataFrame:
    ch = msg["arg"]["channel"]
    if ch.startswith("mark-price-candle"): ktype = "mark-price-candle"
    elif ch.startswith("index-candle"):     ktype = "index-candle"
    else:                                   ktype = "candle"

    return normalize_ws_candles(msg, ktype=ktype)

@register_channel("trades")
def handle_trades(msg: Dict[str, Any]) -> pd.DataFrame:
    return normalize_ws_trades(msg)

@register_channel("books")
def handle_books(msg: Dict[str, Any]) -> pd.DataFrame:
    return normalize_ws_books(msg)


def normalize_rest_candles(rows: List[List[str]]) -> pd.DataFrame:
    # OKX 可能返回倒序，这里统一按 ts 升序
    out = []
    for r in rows:
        ts, o, h, l, c, vol = int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
        out.append({"ts": ts, "dt": _dt(ts), "open": o, "high": h, "low": l, "close": c, "vol": vol})
    return pd.DataFrame(out).sort_values("ts").reset_index(drop=True)

def normalize_ws_candles(msg: Dict[str,Any], ktype: str) -> pd.DataFrame:
    arg = msg.get("arg", {})
    data = msg.get("data", [])
    if not data:
        return pd.DataFrame(columns=["ts","open","high","low","close","aux"])

    rows = []
    for r in data:
        ts = int(r[0])
        o, h, l, c = map(float, r[1:5])
        vol_or_flag = r[5] if len(r) > 5 else None
        rows.append({"ts": ts, "open": o, "high": h, "low": l, "close": c, "aux": vol_or_flag})

    # 仅保留数值列；ts 升序 & 去重（按 ts）
    return (
        pd.DataFrame(rows)
        .sort_values("ts")
        .drop_duplicates(subset=["ts"])
        .reset_index(drop=True)
    )

def normalize_ws_trades(msg: Dict[str,Any]) -> pd.DataFrame:
    arg, data = msg.get("arg", {}), msg.get("data", [])
    inst = arg.get("instId","")
    rows = []
    for r in data:
        ts = int(r["ts"])
        rows.append({
            "instId": inst, "ts": ts, "dt": _dt(ts),
            "px": float(r["px"]), "sz": float(r["sz"]), "side": r["side"],
            "tradeId": r.get("tradeId","")
        })
    return pd.DataFrame(rows).sort_values("ts")

# ---- 订单簿（books/books5）----
def normalize_ws_books(msg: Dict[str,Any]) -> pd.DataFrame:
    arg, data = msg.get("arg", {}), msg.get("data", [])
    if not data: return pd.DataFrame()
    inst = arg.get("instId","")
    snap = data[0]
    ts = int(snap["ts"])
    bids = [(float(p), float(q)) for p,q,*_ in snap.get("bids", [])]
    asks = [(float(p), float(q)) for p,q,*_ in snap.get("asks", [])]
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    return pd.DataFrame([{
        "instId": inst, "ts": ts, "dt": _dt(ts),
        "best_bid": best_bid, "best_ask": best_ask
    }])