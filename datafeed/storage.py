# datafeed/storage.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Iterable, Tuple
import pandas as pd
import re

WritePolicy = Dict[str, Any]
write_policy_registry: Dict[str, WritePolicy] = {}

def register_write_policy(name: str, **policy: Any):
    """
    必填字段：
      - match: Callable[[str], bool]               # 根据 channel 名判断是否命中
      - cols: list[str]                             # 磁盘与内存都要的列（或内存用 mem_cols 覆盖）
      - dedup_on: list[str]                         # 去重键
      - file_kind: Callable[[str, dict], str]       # 文件名用的 kind（与 channel 可不同）
      - mem_key: Callable[[str, dict, str], str]    # 内存键（需包含 inst）
    可选：
      - mem_cols: list[str]                         # 内存只保留的列；缺省用 cols
      - group_by_df_inst: bool                      # 是否按 df.instId 分组，默认 True
    """
    req = ["match", "cols", "dedup_on", "file_kind", "mem_key"]
    for r in req:
        if r not in policy:
            raise ValueError(f"policy '{name}' missing field: {r}")
    write_policy_registry[name] = policy
    return policy

def _resolve_policy(channel: str) -> Tuple[str, WritePolicy]:
    for name, p in write_policy_registry.items():
        if p["match"](channel):
            return name, p
    raise ValueError(f"No write policy matched for channel='{channel}'")

def _sanitize_kind(kind: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", kind)

def _parse_candle(channel: str) -> tuple[str, str]:
    # e.g. "candle1m" -> ("candle","1m")
    #      "mark-price-candle1H" -> ("mark-price-candle","1H")
    #      "index-candle5m" -> ("index-candle","5m")
    for pfx in ("mark-price-candle", "index-candle", "candle"):
        if channel.startswith(pfx):
            bar = channel[len(pfx):] or "1m"
            return pfx, bar
    return "candle", "1m"

# trades
register_write_policy(
    "trades",
    match=lambda ch: ch == "trades",
    cols=["ts","px","sz","side","tradeId"] if True else ["ts","px","sz","side","tradeId"],
    mem_cols=["ts","px","sz","side","tradeId"],           # 内存可不放 instId
    dedup_on=["ts","tradeId"],
    file_kind=lambda ch, arg: "trades",
    mem_key=lambda ch, arg, inst: f"trades:{inst}",
)

# books（按 books 处理：best_bid/best_ask）
register_write_policy(
    "books",
    match=lambda ch: ch.startswith("books"),
    cols=["ts","best_bid","best_ask","bid_sz5", "ask_sz5", "obi_5"],
    mem_cols=["ts","best_bid","best_ask","bid_sz5", "ask_sz5"],
    dedup_on=["ts"],
    file_kind=lambda ch, arg: "books",
    mem_key=lambda ch, arg, inst: f"books:{inst}",
)

# candle / mark-price-candle / index-candle
register_write_policy(
    "candle",
    match=lambda ch: ch.startswith(("candle","mark-price-candle","index-candle")),
    cols=["ts","open","high","low","close","vol","volCcy","volCcyQuote","confirm"],
    dedup_on=["ts"],
    file_kind=lambda ch, arg: (
        lambda ktype, bar: f"candle_{ktype}_{bar}"
    )(*_parse_candle(ch)),
    mem_key=lambda ch, arg, inst: (
        lambda ktype, bar: f"candle:{ktype}:{bar}:{inst}"
    )(*_parse_candle(ch)),
)

# open-interest
register_write_policy(
    "open-interest",
    match=lambda ch: ch.startswith("open-interest"),
    cols=["ts","oi","oiCcy","oiUsd"] if True else ["ts","oi","oiCcy","oiUsd"],
    mem_cols=["ts","oi"],
    dedup_on=["ts"],
    file_kind=lambda ch, arg: "open_interest",
    mem_key=lambda ch, arg, inst: f"open_interest:{inst}",
)

# funding-rate
register_write_policy(
    "funding-rate",
    match=lambda ch: ch.startswith("funding-rate"),
    cols=["ts","fundingRate"] if True else ["ts","fundingRate"],
    mem_cols=["ts","fundingRate"],
    dedup_on=["ts"],
    file_kind=lambda ch, arg: "funding_rate",
    mem_key=lambda ch, arg, inst: f"funding_rate:{inst}",
)

# price-limit
register_write_policy(
    "price-limit",
    match=lambda ch: ch.startswith("price-limit"),
    cols=["ts","buyLmt","sellLmt","enabled"] if True else ["ts","buyLmt","sellLmt","enabled"],
    mem_cols=["ts","buyLmt","sellLmt","enabled"],
    dedup_on=["ts"],
    file_kind=lambda ch, arg: "price_limit",
    mem_key=lambda ch, arg, inst: f"price_limit:{inst}",
)

register_write_policy(
    "mark-price",
    match=lambda ch: ch.startswith("mark-price"),
    cols=["ts","markPx"] if True else ["ts","markPrice"],
    mem_cols=["ts","markPx"],
    dedup_on=["ts"],
    file_kind=lambda ch, arg: "mark_price",
    mem_key=lambda ch, arg, inst: f"mark_price:{inst}",
)

register_write_policy(
    "index-tickers",
    match=lambda ch: ch.startswith("index-tickers"),
    cols=["ts","idxPx"] if True else ["ts","idxPx"],
    mem_cols=["ts","idxPx"],
    dedup_on=["ts"],
    file_kind=lambda ch, arg: "index_tickers",
    mem_key=lambda ch, arg, inst: f"index_tickers:{inst}",
)

register_write_policy(
    "liquidation-orders",
    match=lambda ch: ch.startswith("liquidation-orders"),
    cols=["ts","side","posSide","sz","bkPx","bkLoss"],
    mem_cols=["ts","side","posSide","sz","bkPx","bkLoss"],
    dedup_on=["ts"],
    file_kind=lambda ch, arg: "liquidation_orders",
    mem_key=lambda ch, arg, inst: f"liquidation_orders:{inst}",
)

class MemoryStore:
    def __init__(self):
        self.frame: Dict[str, pd.DataFrame] = {}
        
    def upsert(self, key: str, df: pd.DataFrame, on: str | list[str]):
        if df is None or df.empty: return
        cur = self.frame.get(key)
        if cur is None or cur.empty:
            self.frame[key] = df.drop_duplicates(subset=on).sort_values(on)
        else:
            out = pd.concat([cur, df], ignore_index=True).drop_duplicates(subset=on).sort_values(on)
            self.frame[key] = out
        
    def get(self, key: str) -> pd.DataFrame:
        return self.frame.get(key, pd.DataFrame())

class DiskStore:
    def __init__(self, out_dir: str, backend: str = "csv"):
        self.base = Path(out_dir); self.base.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        if self.backend != "csv":
            raise ValueError("append-only storage currently supports CSV only")
        self._last_ts: dict[str, int] = {}

    def _p(self, kind: str, inst: str) -> Path:
        d = self.base / inst; d.mkdir(parents=True, exist_ok=True)
        return d / f"{kind}.csv"

    def _append_csv(self, path: Path, df: pd.DataFrame):
        header = not path.exists()
        df.to_csv(path, mode="a", header=header, index=False)

    def append(self, kind: str, inst: str, df: pd.DataFrame, cols: list[str], dedup_on: list[str]):
        if df is None or df.empty: return
        key = f"{kind}:{inst}"
        last = self._last_ts.get(key, -1)
        sub = df[df["ts"] > last] if "ts" in df.columns and "ts" in dedup_on else df
        if sub.empty: return
        self._append_csv(self._p(kind, inst), sub[cols])
        if "ts" in sub.columns:
            self._last_ts[key] = int(sub["ts"].max())

    def append_trades(self, inst: str, df: pd.DataFrame):
        if df is None or df.empty: return
        # 轻量去重：仅保留 ts > last_ts 的行
        key = f"trades:{inst}"
        last = self._last_ts.get(key, -1)
        sub = df[df["ts"] > last]
        if sub.empty: return
        self._append_csv(self._p("trades", inst), sub[["ts","px","sz","side","tradeId"]])
        self._last_ts[key] = int(sub["ts"].max())

    def append_books5(self, inst: str, df: pd.DataFrame):
        if df is None or df.empty: return
        key = f"books5:{inst}"
        last = self._last_ts.get(key, -1)
        sub = df[df["ts"] > last]
        if sub.empty: return
        self._append_csv(self._p("books5", inst), sub[["ts","best_bid","best_ask"]])
        self._last_ts[key] = int(sub["ts"].max())

    def append_candle(self, ktype: str, bar: str, inst: str, df: pd.DataFrame):
        if df is None or df.empty: return
        kind = f"candle_{ktype}_{bar}"
        key  = f"{kind}:{inst}"
        last = self._last_ts.get(key, -1)
        sub = df[df["ts"] > last]
        if sub.empty: return
        self._append_csv(self._p(kind, inst), sub[["ts","open","high","low","close","aux"]])
        self._last_ts[key] = int(sub["ts"].max())

    def append_openinterest(self, inst: str, df: pd.DataFrame):
        if df is None or df.empty: return
        key = f"openinterest:{inst}"
        last = self._last_ts.get(key, -1)
        sub = df[df["ts"] > last]
        if sub.empty: return
        self._append_csv(self._p("open_interest", inst), sub[["ts","oi","oiCcy","oiUsd"]])
        self._last_ts[key] = int(sub["ts"].max())

    def append_fundingrate(self, inst: str, df: pd.DataFrame):
        if df is None or df.empty: return
        key = f"fundingrate:{inst}"
        last = self._last_ts.get(key, -1)
        sub = df[df["ts"] > last]
        if sub.empty: return
        self._append_csv(self._p("funding_rate", inst), sub[["ts","fundingRate"]])
        self._last_ts[key] = int(sub["ts"].max())

    


class CompositeStore:
    def __init__(self, mem: MemoryStore, disk: DiskStore):
        self.mem = mem
        self.disk = disk

    def _iter_inst_groups(self, df: pd.DataFrame, arg: dict, use_df_group: bool) -> Iterable[Tuple[str, pd.DataFrame]]:
        if use_df_group and "instId" in df.columns:
            for inst, sub in df.groupby("instId"):
                yield inst, sub
        else:
            inst = arg.get("instId", "UNKNOWN")
            yield inst, df


    def write(self, channel: str, arg: dict, df: pd.DataFrame):
        """
        通用写入：仅依赖 channel name；其余逻辑由策略表决定。
        """
        _, policy = _resolve_policy(channel)
        cols = policy["cols"]
        mem_cols = policy.get("mem_cols", cols)
        dedup_on = policy["dedup_on"]
        file_kind_fn: Callable[[str, dict], str] = policy["file_kind"]
        mem_key_fn:  Callable[[str, dict, str], str] = policy["mem_key"]
        group_by_df_inst = policy.get("group_by_df_inst", True)

        kind = file_kind_fn(channel, arg)
        for inst, sub in self._iter_inst_groups(df, arg, use_df_group=group_by_df_inst):
            mem_key = mem_key_fn(channel, arg, inst)
            self.mem.upsert(mem_key, sub[mem_cols], on=dedup_on)
            self.disk.append(kind, inst, sub[cols], cols, dedup_on)

    def write_candle(self, inst: str, ktype: str, bar: str, df: pd.DataFrame):
        """
        文件结构： out_dir/{inst}/candle_{ktype}_{bar}.csv
        列：ts,open,high,low,close,aux
        """
        if df is None or df.empty:
            return
        # 内存键按 (inst, ktype, bar) 组织
        key = f"candle:{ktype}:{bar}:{inst}"
        # 只按 ts 去重
        self.mem.upsert(key, df[["ts","open","high","low","close","aux"]], on=["ts"])
        # 磁盘：走 append（见下文 DiskStore.append_candle）
        self.disk.append_candle(ktype, bar, inst, df[["ts","open","high","low","close","aux"]])
    
    def write_trades(self, df: pd.DataFrame):
        if df is None or df.empty: return
        for inst, sub in df.groupby("instId"):
            self.mem.upsert(f"trades:{inst}", sub[["ts","px","sz","side","tradeId"]], on=["ts","tradeId"])
            self.disk.append_trades(inst, sub)

    def write_books(self, df: pd.DataFrame):
        if df is None or df.empty: return
        for inst, sub in df.groupby("instId"):
            self.mem.upsert(f"books5:{inst}", sub[["ts","best_bid","best_ask"]], on=["ts"])
            self.disk.append_books5(inst, sub)

    def write_openinterest(self, df: pd.DataFrame):
        if df is None or df.empty: return
        for inst, sub in df.groupby("instId"):
            self.mem.upsert(f"open_interest:{inst}", sub[["ts","oi"]], on=["ts"])
            self.disk.append_openinterest(inst, sub)
        
    def write_fundingrate(self, df: pd.DataFrame):
        if df is None or df.empty: return
        for inst, sub in df.groupby("instId"):
            self.mem.upsert(f"funding_rate:{inst}", sub[["ts","fundingRate"]], on=["ts"])
            self.disk.append_fundingrate(inst, sub)
