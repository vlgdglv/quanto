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

    arg = msg.get("arg", {})
    inst = arg.get("instId","")
    data = msg.get("data", [])
    if not data:
        return pd.DataFrame(columns=["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"])

    rows = []
    for r in data:
        ts = int(r[0])
        o, h, l, c = map(float, r[1:5])
        vol = r[5] if len(r) > 5 else None
        volCcy, volCcyQuote = float(r[6]) if len(r) > 6 else None, float(r[7]) if len(r) > 7 else None
        confirm = r[8] if len(r) > 8 else None
        rows.append({"ts": ts, "open": o, "high": h, "low": l, "close": c, 
                     "vol": vol, "volCcy": volCcy, "volCcyQuote": volCcyQuote, 
                     "confirm": confirm})

    return (
        pd.DataFrame(rows)
        .sort_values("ts")
        .drop_duplicates(subset=["ts"])
        .reset_index(drop=True)
    )

@register_channel("trades")
def handle_trades(msg: Dict[str, Any]) -> pd.DataFrame:
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

@register_channel("books")
def handle_books(msg: Dict[str, Any]) -> pd.DataFrame:
    arg, data = msg.get("arg", {}), msg.get("data", [])
    if not data: return pd.DataFrame()
    inst = arg.get("instId","")
    snap = data[0]
    ts = int(snap["ts"])
    bids = [(float(p), float(q)) for p,q,*_ in snap.get("bids", [])]
    asks = [(float(p), float(q)) for p,q,*_ in snap.get("asks", [])]
    best_bid = bids[0][0] if bids else None
    best_ask = asks[0][0] if asks else None
    bid_sz5 = sum(q for _, q in bids[:5]) if bids else None
    ask_sz5 = sum(q for _, q in asks[:5]) if asks else None
    obi_5 = None
    if bid_sz5 and ask_sz5:
        denom = (bid_sz5 or 0) + (ask_sz5 or 0)
        obi_5 = ((bid_sz5 - ask_sz5) / denom) if denom and denom > 0 else 0.0
    return pd.DataFrame([{
        "instId": inst, "ts": ts, "dt": _dt(ts),
        "best_bid": best_bid, "best_ask": best_ask,
        "bid_sz5": bid_sz5, "ask_sz5": ask_sz5,
        "obi_5": obi_5
    }])

@register_channel("open-interest")
def handle_open_interest(msg: Dict[str, Any]) -> pd.DataFrame:
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

@register_channel("funding-rate")
def handle_funding_rate(msg: Dict[str, Any]) -> pd.DataFrame:
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

@register_channel("mark-price")
def handle_mark_price(msg: Dict[str, Any]) -> pd.DataFrame:
    data = msg.get("data", [])
    if not data:
        return pd.DataFrame(columns=["ts","markPx"])
    rows = [{"ts": _to_int_ms(d.get("ts")), "markPx": _to_float(d.get("markPx"))} for d in data]
    return (pd.DataFrame(rows).dropna(subset=["ts"])
            .sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True))

@register_channel("index-tickers")
def handle_index_tickers(msg: Dict[str, Any]) -> pd.DataFrame:
    data = msg.get("data", [])
    if not data:
        return pd.DataFrame(columns=["ts","idxPx"])
    rows = [{"ts": _to_int_ms(d.get("ts")), "idxPx": _to_float(d.get("idxPx"))} for d in data]
    return (pd.DataFrame(rows).dropna(subset=["ts"])
            .sort_values("ts").drop_duplicates(subset=["ts"]).reset_index(drop=True))

@register_channel("liquidation-orders")
def handle_liquidations(msg: Dict[str, Any]) -> pd.DataFrame:
    """
    注意：该频道每合约每秒最多一条，不代表全量清算；fields 见官方示例。
    """
    data = msg.get("data", [])
    if not data:
        return pd.DataFrame(columns=[
            "ts","instId","instFamily","uly","side","posSide","sz","bkPx","bkLoss"
        ])
    rows: List[Dict[str, Any]] = []
    for d in data:
        base = {
            "instId": d.get("instId"),
            "instFamily": d.get("instFamily"),
            "uly": d.get("uly"),
        }
        for it in d.get("details", []) or []:
            rows.append({
                **base,
                "ts": _to_int_ms(it.get("ts")),
                "side": it.get("side"),
                "posSide": it.get("posSide"),
                "sz": _to_float(it.get("sz")),
                "bkPx": _to_float(it.get("bkPx")),
                "bkLoss": _to_float(it.get("bkLoss")),
            })
    return (
        pd.DataFrame(rows)
        .dropna(subset=["ts"])
        .sort_values(["ts","instId"])
        .drop_duplicates(subset=["ts","instId","side","posSide","sz"], keep="last")
        .reset_index(drop=True)
    )

@register_channel("price-limit")
def handle_price_limit(msg: Dict[str, Any]) -> pd.DataFrame:
    """
    WS channel: 'price-limit'
    Push fields: buyLmt, sellLmt, enabled, ts（另含 instId/instType）
    Docs: OKX V5 WebSocket / Price limit channel
    """
    data = msg.get("data", [])
    if not data:
        return pd.DataFrame(columns=["ts", "buyLmt", "sellLmt", "enabled"])

    rows: List[Dict[str, Any]] = []
    for d in data:
        rows.append({
            "ts": _to_int_ms(d.get("ts")),
            "buyLmt": _to_float(d.get("buyLmt")),
            "sellLmt": _to_float(d.get("sellLmt")),
            # enabled 为 bool；转为 1/0 以保持“仅数值列”风格
            "enabled": int(bool(d.get("enabled"))) if d.get("enabled") is not None else None,
        })

    return (
        pd.DataFrame(rows)
        .dropna(subset=["ts"])
        .sort_values("ts")
        .drop_duplicates(subset=["ts"])
        .reset_index(drop=True)
    )