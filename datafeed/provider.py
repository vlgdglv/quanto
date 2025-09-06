# datafeed/provider_okx.py
from typing import Dict, Any, Iterable, List, Set, DefaultDict
from collections import defaultdict
import re

# def build_ws_url(cfg: dict) -> str:
#     mode = cfg["env"]["mode"]  # paper | live
#     ws_cfg = cfg["okx"]["ws"][mode]
#     use_business = cfg["datafeed"].get("use_business", False)
#     if use_business:
#         return ws_cfg["business"]
#     # 默认公共
#     return ws_cfg.get("public", ws_cfg.get("business"))

def _index_from_inst(inst: str) -> str:
    """
    把合约/期货/期权的 instId 还原成指数ID（uly）：
      ETH-USDT-SWAP        -> ETH-USDT
      BTC-USD-240927       -> BTC-USD
      BTC-USD-240927-40000-C  -> BTC-USD  (期权)
      以及已经是指数ID的 ETH-USDT 保持不变
    """
    parts = inst.split("-")
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return inst

_inst_type_pat_futures = re.compile(r"-\d{6,8}$") 

def _inst_type_from_inst(inst: str) -> str:
    """
    从 instId 粗略推断 instType：SWAP / FUTURES / OPTION / SPOT
    （若 channel_cfg 显式给了 instType(s)，优先用配置）
    """
    if inst.endswith("-SWAP"):
        return "SWAP"
    if _inst_type_pat_futures.search(inst):
        return "FUTURES"
    # 简单判断期权：...-strike-(C|P)
    parts = inst.split("-")
    if len(parts) >= 5 and parts[-1] in ("C", "P"):
        return "OPTION"
    # 两段形如 BTC-USDT 基本可视作现货/杠杆（MARGIN）
    if inst.count("-") == 1:
        return "MARGIN"
    return "ANY"

def build_ws_url(cfg: dict, ws_kind: str) -> str:
    mode = cfg["datafeed"]["mode"]  # paper | live
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

def _build_market_price_args(channel_cfg: Dict[str, Any], insts: List[str]) -> List[Dict[str, Any]]:
    if not channel_cfg.get("fetch", False):
        return []
    return [{"channel": "mark-price", "instId": inst} for inst in insts]

def _build_funding_rate_args(channel_cfg: Dict[str, Any], insts: List[str]) -> List[Dict[str, Any]]:
    if not channel_cfg.get("fetch", False):
        return []
    return [{"channel": "funding-rate", "instId": inst} for inst in insts]

def _build_open_interest_args(channel_cfg: Dict[str, Any], insts: List[str]) -> List[Dict[str, Any]]:
    if not channel_cfg.get("fetch", False):
        return []
    return [{"channel": "open-interest", "instId": inst} for inst in insts]

def _build_index_tickers_args(channel_cfg: Dict[str, Any], insts: List[str]) -> List[Dict[str, Any]]:
    """
    指数频道 index-tickers 必须订阅“指数ID”（与 uly 相同），
    不能用合约ID（不能带 -SWAP / -YYYYMMDD 等尾缀）。
    这里对传入 insts 做归一化与去重。
    """
    if not channel_cfg.get("fetch", False):
        return []

    # 允许用户传 ETH-USDT-SWAP 或 BTC-USD-240927，我们统一映射为 ETH-USDT / BTC-USD
    idx_ids: List[str] = [_index_from_inst(x) for x in insts]
    # 去重并保持原有顺序
    seen: Set[str] = set()
    deduped = []
    for x in idx_ids:
        if x not in seen:
            seen.add(x)
            deduped.append(x)

    return [{"channel": "index-tickers", "instId": idx} for idx in deduped]


def _build_liquidation_orders_args(channel_cfg: Dict[str, Any], insts: List[str]) -> List[Dict[str, Any]]:
    """
    清算频道 liquidation-orders 只接受 instType，不接受 instId。
    OKX 会按 instType 推送该类型下所有合约的清算快照（每合约每秒最多一条）。
    这里支持三种来源：
      1) channel_cfg["instTypes"] = ["SWAP","FUTURES",...]  显式指定（优先）
      2) channel_cfg["instType"] = "SWAP"                  单个指定
      3) 由 insts 自动推断出类型集合（如传入若干 instId）
    """
    if not channel_cfg.get("fetch", False):
        return []

    # 优先使用配置覆盖
    if "instTypes" in channel_cfg and channel_cfg["instTypes"]:
        types = list(dict.fromkeys([t.upper() for t in channel_cfg["instTypes"]]))
    elif "instType" in channel_cfg and channel_cfg["instType"]:
        types = [channel_cfg["instType"].upper()]
    else:
        # 从 insts 猜测类型集合
        guessed = [ _inst_type_from_inst(x) for x in insts ]
        # 只保留 OKX 文档允许的四类
        allow = {"SWAP","FUTURES","MARGIN","OPTION"}
        types = [t for t in dict.fromkeys(guessed) if t in allow]
        # 如果完全猜不到，就默认订 SWAP（常见）
        if not types:
            types = ["SWAP"]

    return [{"channel": "liquidation-orders", "instType": t} for t in types]


def _build_price_limit_args(channel_cfg: Dict[str, Any], insts: List[str]) -> List[Dict[str, Any]]:
    if not channel_cfg.get("fetch", False):
        return []
    return [{"channel": "price-limit", "instId": inst} for inst in insts]


_channel_builders = {
    "candles": _build_candle_args,
    "trades": _build_trade_args,
    "books": _build_book_args,
    "funding-rate": _build_funding_rate_args,
    "open-interest": _build_open_interest_args,
    "mark-price": _build_market_price_args,
    "index-tickers": _build_index_tickers_args,
    "liquidation-orders": _build_liquidation_orders_args,
    "price-limit": _build_price_limit_args,
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
        "funding-rate": "public",
        "open-interest": "public",
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