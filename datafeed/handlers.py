# data/normalizer.py
from datetime import datetime, timezone
from typing import Dict, Any, Callable, List, Optional
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

def _to_float(x: Optional[str]) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None
    
def _to_int_ms(x: Optional[str]) -> Optional[int]:
    if x is None or x == "":
        return None
    try:
        return int(x)
    except Exception:
        return None

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

@register_channel("open-interest")
def handle_open_interest(msg: Dict[str, Any]) -> pd.DataFrame:
    return normalize_ws_open_interest(msg)

@register_channel("funding-rate")
def handle_funding_rate(msg: Dict[str, Any]) -> pd.DataFrame:
    return normalize_ws_funding_rate(msg)

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

def normalize_ws_open_interest(msg: Dict[str,Any]) -> pd.DataFrame:
    """
    OKX WS channel: 'open-interest'
    Push example & fields: oi, oiCcy, oiUsd, ts, instId, instType
    Docs: https://app.okx.com/docs-v5/en/  -> Open interest channel
    """
    data = msg.get("data", [])
    if not data:
        return pd.DataFrame(columns=["ts","oi","oiCcy","oiUsd","instId"])
    
    rows: List[Dict[str, Any]] = []
    for d in data:
        rows.append({
            "ts": _to_int_ms(d.get("ts")),
            "oi": _to_float(d.get("oi")),           # 合约张数
            "oiCcy": _to_float(d.get("oiCcy")),     # 标的计价（如 BTC）数量
            "oiUsd": _to_float(d.get("oiUsd")),     # USD 估值
            "instId": d.get("instId"),
        })
    
    return (
        pd.DataFrame(rows)
            .dropna(subset=["ts"])
            .sort_values("ts")
            .drop_duplicates(subset=["ts"])
            .reset_index(drop=True)
    )

def normalize_ws_funding_rate(msg: Dict[str,Any]) -> pd.DataFrame:
    """
    OKX WS channel: 'funding-rate'
    Push fields (精简常用数值 + 时间戳)：
      fundingRate, premium, minFundingRate, maxFundingRate,
      fundingTime, nextFundingTime, ts
    其余文本状态（如 settState）可按需扩展。
    Docs: https://app.okx.com/docs-v5/en/  -> Funding rate channel
    备注：推送里同时出现 ts（数据返回时间）与 fundingTime（结算时间）。
    这里以 ts 作为主时间轴；另保留 fundingTime/nextFundingTime 供对齐结算周期。
    """
    data = msg.get("data", [])
    if not data:
        return pd.DataFrame(
            columns=[
                "ts", "fundingRate", "premium",
                "minFundingRate", "maxFundingRate",
                "fundingTime", "nextFundingTime", "instId"
            ]
        )

    rows: List[Dict[str, Any]] = []
    for d in data:
        rows.append({
            "ts": _to_int_ms(d.get("ts")) or _to_int_ms(d.get("fundingTime")),
            "fundingRate": _to_float(d.get("fundingRate")),
            "premium": _to_float(d.get("premium")),
            "minFundingRate": _to_float(d.get("minFundingRate")),
            "maxFundingRate": _to_float(d.get("maxFundingRate")),
            "fundingTime": _to_int_ms(d.get("fundingTime")),
            "nextFundingTime": _to_int_ms(d.get("nextFundingTime")),
            "instId": d.get("instId"),
        })

    return (
        pd.DataFrame(rows)
        .dropna(subset=["ts"])
        .sort_values("ts")
        .drop_duplicates(subset=["ts"])
        .reset_index(drop=True)
    )