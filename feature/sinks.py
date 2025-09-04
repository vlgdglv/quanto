# feature/sinks.py
from __future__ import annotations
import os, sqlite3, re
from abc import ABC, abstractmethod
from typing import Optional, Callable, Sequence, Tuple
import pandas as pd

class FeatureSink(ABC):
    """落库抽象接口——方便后续扩展到 Parquet、Kafka、ClickHouse 等。"""
    @abstractmethod
    def write(self, df: pd.DataFrame) -> None:
        """同步写入一批特征。要求是幂等/可重入（失败可重试）。"""
        ...

    @abstractmethod
    def close(self) -> None:
        """释放资源（可选覆盖）。"""
        ...

class CSVFeatureSink(FeatureSink):
    def __init__(self, path: str, 
                 mode: str = "a", by: Optional[str] = None,
                 key_fn: Optional[Callable[[str], str]] = None,
                 path_template: Optional[str] = None):
        """
        :param path: 目标 CSV 路径（单文件，包含 instId/tf/ts 列，不做分表）
        :param mode: 'a' 追加 / 'w' 覆盖
        :param by:   分文件的列名，比如 'instId' 或 'instType'；None 表示不分文件
        :param key_fn: 对分组键做转换的函数（如从 'BTC-USDT-SWAP' 提取 'BTC'）
        :param path_template: 自定义分文件路径模板，包含 {key} 占位符
                              例: "data/features-{key}.csv"
        """
        self.path = path
        self.mode = mode
        self.by = by
        self.key_fn = key_fn or (lambda x: x)
        self.path_template = path_template
        self._header_written_map: dict[str, bool] = {}
        self._header_written_map[path] = os.path.exists(path) and mode == "a"

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    @staticmethod
    def _sanitize_key(val: object) -> str:
        s = str(val)
        return re.sub(r'[^A-Za-z0-9_\-]', "_", s)
    
    def _resolve_path(self, key_val: object) -> str:
        safe_key = self._sanitize_key(self.key_fn(key_val))
        if self.path_template:
            return self.path_template.format(key=safe_key)
        d = os.path.dirname(self.path) or "."
        base = os.path.basename(self.path)
        stem, ext = os.path.splitext(base)
        if not ext:
            ext = ".csv"
        return os.path.join(d, f"{stem}-{safe_key}{ext}")

    def _write_one(self, path: str, df: pd.DataFrame) -> None:
        cols = list(df.columns)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        header_written = self._header_written_map.get(path, os.path.exists(path) and self.mode == "a")
        write_header = not header_written and (self.mode != "a" or not os.path.exists(path))
        
        df.to_csv(path, 
                  mode=self.mode if os.path.exists(path) else "w",
                  index=False, 
                  header=write_header)
        self._header_written_map[path] = True

    def write(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        
        if not self.by or self.by not in df.columns:
            self._write_one(self.path, df)
            self.mode = "a"
            return

        for key, group in df.groupby(self.by, dropna=False):
            path = self._resolve_path(key)
            self._write_one(path, group)

        self.mode = "a"

    def close(self) -> None:
        pass

class SQLiteFeatureSink(FeatureSink):
    """
    轻量 SQLite 落库；不用 SQLAlchemy，直接 sqlite3.executemany。
    要求唯一键：(instId, tf, ts) 去重。
    """
    def __init__(self, db_path: str, table: str = "features", columns: Sequence[str] = ()):
        self.db_path = db_path
        self.table = table
        self.columns = list(columns) if columns else []
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        if self.columns:
            self._ensure_table()

    def _ensure_table(self):
        # 简单推断：前三列是 instId/tf/ts，其他按 REAL 存（也可从 cfg 显式传类型）
        cols_def = []
        for i, c in enumerate(self.columns):
            if c in ("instId", "tf"):
                cols_def.append(f'"{c}" TEXT NOT NULL')
            elif c == "ts":
                cols_def.append(f'"{c}" INTEGER NOT NULL')
            else:
                cols_def.append(f'"{c}" REAL')
        cols_sql = ",\n  ".join(cols_def)
        # 唯一键用于去重
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS "{self.table}" (
          {cols_sql},
          UNIQUE("instId","tf","ts") ON CONFLICT REPLACE
        );
        """
        self.conn.execute(create_sql)
        self.conn.commit()

    def write(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        if not self.columns:
            self.columns = list(df.columns)
            self._ensure_table()
        # 保证列顺序
        df = df[self.columns].copy()
        placeholders = ",".join(["?"] * len(self.columns))
        sql = f'INSERT OR REPLACE INTO "{self.table}" ({",".join(self.columns)}) VALUES ({placeholders})'
        data = list(map(tuple, df.itertuples(index=False, name=None)))
        with self.conn:  # 自动事务
            self.conn.executemany(sql, data)

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
