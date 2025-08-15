# datafeed/provider_okx.py
from typing import Dict, Any, Iterable, List, Tuple, DefaultDict
from collections import defaultdict

# def build_ws_url(cfg: dict) -> str:
#     mode = cfg["env"]["mode"]  # paper | live
#     ws_cfg = cfg["okx"]["ws"][mode]
#     use_business = cfg["datafeed"].get("use_business", False)
#     if use_business:
#         return ws_cfg["business"]
#     # 默认公共
#     return ws_cfg.get("public", ws_cfg.get("business"))


def build_ws_url(cfg: dict, ws_kind: str) -> str:
    mode = cfg["env"]["mode"]  # paper | live
    ws_cfg = cfg["okx"]["ws"][mode]
    return ws_cfg.get(ws_kind, ws_cfg.get("public", ws_cfg.get("business")))


def _build_candle_args(channel_cfg: Dict[str, Any], insts: List[str]) -> List[Dict[str, Any]]:
    if not channel_cfg.get("fetch", False):
        return []
    kind = channel_cfg.get("kind", "trade")  # trade | mark | index
    bars = channel_cfg.get("bars", ["1m"])
    prefix_map = {
        "trade": "candle",
        "mark": "mark-price-candle",
        "index": "index-candle",
    }
    prefix = prefix_map.get(kind, "candle")
    return [{"channel": f"{prefix}{bar}", "instId": inst} for inst in insts for bar in bars]

def _build_trade_args(channel_cfg: Dict[str, Any], insts: List[str]) -> List[Dict[str, Any]]:
    if not channel_cfg.get("fetch", False):
        return []
    return [{"channel": "trades", "instId": inst} for inst in insts]

def _build_book_args(channel_cfg: Dict[str, Any], insts: List[str]) -> List[Dict[str, Any]]:
    if not channel_cfg.get("fetch", False):
        return []
    level = channel_cfg.get("level", 400)
    
    if level == 400:
        ch = "books"
    elif level == 5:
        ch = "books5"
    elif level == 1:
        ch = "bbo-tbt"
    else:
        ch = "books"
    return [{"channel": ch, "instId": inst} for inst in insts]

def _build_marketprice_args(channel_cfg: Dict[str, Any], insts: List[str]) -> List[Dict[str, Any]]:
    if not channel_cfg.get("fetch", False):
        return []
    return [{"channel": "mark-price", "instId": inst} for inst in insts]

def _build_fundingrate_args(channel_cfg: Dict[str, Any], insts: List[str]) -> List[Dict[str, Any]]:
    if not channel_cfg.get("fetch", False):
        return []
    return [{"channel": "funding-rate", "instId": inst} for inst in insts]

_channel_builders = {
    "candles": _build_candle_args,
    "trades": _build_trade_args,
    "books": _build_book_args,
    "mark-price": _build_marketprice_args,
    "funding-rate": _build_fundingrate_args

}

def build_subscribe_args(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    datafeed = cfg.get("datafeed", {})
    insts = datafeed.get("instIds", [])
    channel_args = datafeed.get("channels", [])
    args: List[Dict[str, Any]] = []

    for ch_name, builder in _channel_builders.items():
        ch_cfg = channel_args.get(ch_name, {})
        args.extend(builder(ch_cfg, insts))

    for ch in datafeed.get("extra_channels", []):
        for inst in insts:
            args.append({"channel": ch, "instId": inst})

    return args


def _ws_kind_for_channle(cfg: Dict[str, Any], category: str) -> str:
    # 默认路由：c/t/b -> public；mp/fr -> business
    defaults = {
        "candles": "public",
        "trades": "public",
        "books": "public",
        "mark-price": "business",
        "funding-rate": "business",
    }
    override = cfg.get("datafeed", {}).get("ws_kind_per_channel", {})
    return override.get(category, defaults.get(category, "public"))

def build_ws_plan(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    返回一个极简 plan 列表，每个元素含：url、args、ws_kind
    例如：[
      {"ws_kind":"public", "url":"wss://.../public", "args":[...]},
      {"ws_kind":"business","url":"wss://.../business","args":[...]}
    ]
    """
    datafeed = cfg.get("datafeed", {})
    insts = datafeed.get("instIds", [])
    ch_cfgs = datafeed.get("channels", {})

    # 先按“大类 channel”构建各自的 args
    channel_args: Dict[str, List[Dict[str, Any]]] = {}
    for channel, builder in _channel_builders.items():
        cfg_i = ch_cfgs.get(channel, {})
        channel_args[channel] = builder(cfg_i, insts)

    # 额外频道（若有）默认归到 public（你也可以自己扩展，但尽量少动）
    extra = []
    for ch in datafeed.get("extra_channels", []):
        for inst in insts:
            extra.append({"channel": ch, "instId": inst})
    if extra:
        channel_args.setdefault("extra", []).extend(extra)

    # 将每个“大类 channel”的 args 按其 ws_kind 汇总
    grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for channel, args in channel_args.items():
        if not args:
            continue
        ws_kind = _ws_kind_for_channle(cfg, channel if channel in _channel_builders else "candles")
        grouped[ws_kind].extend(args)

    # 产出 plan
    plan: List[Dict[str, Any]] = []
    for ws_kind, args in grouped.items():
        if not args:
            continue
        plan.append({
            "ws_kind": ws_kind,
            "url": build_ws_url(cfg, ws_kind),
            "args": args
        })
    return plan