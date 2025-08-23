# feature/sinks.py
from __future__ import annotations
import os, sqlite3, time
from abc import ABC, abstractmethod
from typing import Iterable, List, Sequence, Tuple
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
    def __init__(self, path: str, mode: str = "a"):
        """
        :param path: 目标 CSV 路径（单文件，包含 instId/tf/ts 列，不做分表）
        :param mode: 'a' 追加 / 'w' 覆盖
        """
        self.path = path
        self.mode = mode
        self._header_written = os.path.exists(path) and mode == "a"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        # 统一列顺序（若调用者未保证）
        cols = list(df.columns)
        # 如果是新文件且不是追加，则认为需要写表头
        write_header = not self._header_written and (self.mode != "a" or not os.path.exists(self.path))
        df.to_csv(self.path, mode=self.mode if os.path.exists(self.path) else "w",
                  index=False, header=write_header)
        self._header_written = True
        # 后续都用追加
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
