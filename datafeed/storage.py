# datafeed/storage.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import pandas as pd



# 你可以将它放在 storage.py 顶部
write_policy_registry: Dict[str, Dict[str, Any]] = {
    "trades": {
        "match": lambda kind: kind == "trades",
        "cols": ["ts", "px", "sz", "side", "tradeId"],
        "on": ["ts", "tradeId"]
    },
    "books": {
        "match": lambda kind: kind.startswith("books"),
        "cols": ["ts", "best_bid", "best_ask"],
        "on": ["ts"]
    },
    "candle": {
        "match": lambda kind: kind.startswith(("candle", "mark-price-candle", "index-candle")),
        "parse": lambda kind: {
            "ktype": kind.split("_")[1],
            "bar": "_".join(kind.split("_")[2:]),
        },
        "cols": ["ts", "open", "high", "low", "close", "aux"],
        "on": ["ts"],
    }
}

def _resolve_policy(kind: str) -> tuple[str, dict]:
    for name, policy in write_policy_registry.items():
        if policy["match"](kind):
            return name, policy
    raise ValueError(f"No write policy matched for kind='{kind}'")


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
        # 仅 CSV 支持无锁快速 append；parquet 仍建议批量写
        if self.backend != "csv":
            raise ValueError("append-only storage currently supports CSV only")
        # 记录每个文件的最近 ts，防止回放/重连小抖动导致重复写
        self._last_ts: dict[str, int] = {}

    def _p(self, kind: str, inst: str) -> Path:
        d = self.base / inst; d.mkdir(parents=True, exist_ok=True)
        return d / f"{kind}.csv"

    def _append_csv(self, path: Path, df: pd.DataFrame):
        # 首次写入带 header；后续仅写数据行
        header = not path.exists()
        # 追加
        df.to_csv(path, mode="a", header=header, index=False)

    def append(self, kind: str, inst: str, df: pd.DataFrame, cols: list[str], on: list[str]):
        if df is None or df.empty: return
        key = f"{kind}:{inst}"
        last = self._last_ts.get(key, -1)
        sub = df[df["ts"] > last] if "ts" in df.columns else df
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

    


class CompositeStore:
    def __init__(self, mem: MemoryStore, disk: DiskStore):
        self.mem = mem
        self.disk = disk

    # def write_df(self, kind: str, df: pd.DataFrame):
    #     if df is None or df.empty: return
    #     base_kind, policy = _resolve_policy(kind)
    #     cols = policy["cols"]
    #     on = policy["on"]
    #     parser = policy.get("parse", lambda _: {})  # default empty

    #     for inst, sub in df.groupby("instId"):
    #         key = f"{kind}:{inst}"
    #         self.mem.upsert(key, sub[cols], on=on)

    #         # 解析实际磁盘 kind（仅 candle 用）
    #         meta = parser(kind)
    #         disk_kind = kind if base_kind != "candle" else f"candle_{meta['ktype']}_{meta['bar']}"

    #         self.disk.append(disk_kind, inst, sub, cols=cols, on=on)

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
            # 内存仍做全量（便于实时读取），磁盘只 append
            self.mem.upsert(f"trades:{inst}", sub[["ts","px","sz","side","tradeId"]], on=["ts","tradeId"])
            self.disk.append_trades(inst, sub)

    def write_books(self, df: pd.DataFrame):
        if df is None or df.empty: return
        for inst, sub in df.groupby("instId"):
            self.mem.upsert(f"books5:{inst}", sub[["ts","best_bid","best_ask"]], on=["ts"])
            self.disk.append_books5(inst, sub)

    # # Trades
    # def write_trades(self, df: pd.DataFrame):
    #     for inst, sub in df.groupby("instId"):
    #         key = f"trades:{inst}"
    #         self.mem.upsert(key, sub, on=["ts","tradeId"])
    #         self.disk.upsert(kind="trades", inst=inst, df=sub, on=["ts","tradeId"])

    # # Books（简化只写 best_bid/ask）
    # def write_books(self, df: pd.DataFrame):
    #     for inst, sub in df.groupby("instId"):
    #         key = f"books5:{inst}"
    #         self.mem.upsert(key, sub, on=["ts"])
    #         self.disk.upsert(kind="books5", inst=inst, df=sub, on=["ts"])