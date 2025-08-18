# feature/integrate.py
from typing import Dict, Any, Optional
import re
import pandas as pd
from datafeed.handlers import channel_registry  # 你给的 register_channel 映射
from feature.engine_pd import FeatureEnginePD

_tf_pat = re.compile(r"(?:candle)(\d+(?:ms|s|m|h|d))")

def _extract_tf(channel: str) -> Optional[str]:
    m = _tf_pat.search(channel)
    return m.group(1) if m else None

def process_msg(msg: Dict[str, Any], engine: FeatureEnginePD) -> Optional[pd.DataFrame]:
    """
    输入：原始 WS 消息（含 arg/data），用你现有的 handler 解析后喂引擎
    返回：当且仅当是“bar 收盘”产生特征时，返回 features DataFrame；否则返回 None
    """
    arg = msg.get("arg", {})
    channel = arg.get("channel","")
    instId  = arg.get("instId","")
    prefix = None
    for k in channel_registry.keys():
        if channel.startswith(k):
            prefix = k; break
    if not prefix:
        return None

    df = channel_registry[prefix](msg)
    if df is None or df.empty:
        return None

    if prefix in ("candle", "mark-price-candle", "index-candle"):
        tf = _extract_tf(channel) or "1m"
        feats = engine.update_candles(df, instId=instId, tf=tf)
        return feats if not feats.empty else None
    elif prefix == "books":
        engine.update_books(df, instId=instId, tf="1m")
        return None
    elif prefix == "trades":
        engine.update_trades(df, instId=instId, tf="1m")
        return None
    else:
       return None
