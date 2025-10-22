# apps/run_consumer_feature_writer.py
"""

conda activate py312-crypto
python -m app.run_consumer_feature_writer --streams BTC-USDT-SWAP ETC-USDT-SWAP DOGE-USDT-SWAP

"""
import os, asyncio, json, time, signal, contextlib, argparse
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime
import csv
import pyarrow as pa
import pyarrow.parquet as pq

from infra.redis_stream import RedisStreamsSubscriber
from utils.logger import logger


# ========== 配置 ==========
BASE_DIR      = Path(os.getenv("BASE_DIR", "data/features_csv"))
REDIS_DSN     = os.getenv("REDIS_DSN", "redis://:Your$Pass@127.0.0.1:6379/0")
STREAM_NAME   = os.getenv("REDIS_STREAM", "features")
START_POS     = os.getenv("START", "now")
CONCURRENCY   = int(os.getenv("CONCURRENCY", "1"))

# flush 策略（任选其一或同时满足）
FLUSH_ROWS    = int(os.getenv("FLUSH_ROWS", "100"))     # 一批行数达到即 flush
FLUSH_SEC     = float(os.getenv("FLUSH_SEC", "60.0"))    # 距上次 flush 达到该秒数即 flush


def env_default(name: str, default=None):
    return os.getenv(name, default)

def build_parser():
    p = argparse.ArgumentParser("features-parquet-consumer")
    p.add_argument("--redis-dsn", default=env_default("REDIS_DSN", "redis://:12345678@127.0.0.1:6379/0"))
    p.add_argument("--streams", nargs="+", default=None, help="list of redis stream names, e.g., features:BTC-USDT-SWAP features:DOGE-USDT-SWAP")
    p.add_argument("--stream",    default=env_default("REDIS_STREAM", None))
    p.add_argument("--start",     default=env_default("START", "now"), choices=["now","earliest"])
    p.add_argument("--base-dir", default=env_default("BASE_DIR", "data/features_csv"))
    p.add_argument("--flush-rows", type=int, default=int(env_default("FLUSH_ROWS", "500")))
    p.add_argument("--flush-sec",  type=float, default=float(env_default("FLUSH_SEC", "2.0")))
    p.add_argument("--concurrency", type=int, default=int(env_default("CONCURRENCY", "1")))
    return p


# ========== Parquet Sinker ==========
class ParquetSink:
    """
    以 (inst, tf) 为粒度写入：
      <BASE_DIR>/<inst>/<tf>/data-<runid>.parquet
    单进程内保持 ParquetWriter 打开，按批写入 row group。
    """
    def __init__(self, base_dir: Path, flush_rows: int = 500, flush_sec: float = 2.0, stream_name: str = None):
        self.base_dir = base_dir
        self.flush_rows = flush_rows
        self.flush_sec = flush_sec

        self._buffers: Dict[Tuple[str, str], List[Dict]] = {}
        self._writers: Dict[Tuple[str, str], pq.ParquetWriter] = {}
        self._schemas: Dict[Tuple[str, str], pa.Schema] = {}
        self._files: Dict[Tuple[str, str], Path] = {}
        self._last_flush_mono = time.monotonic()
        # 为了“不分片且文件名固定在一次运行内”，给本轮运行生成一个 runid
        # self._runid = str(int(time.time()))
        self._runid = datetime.now().strftime("%y%m%d_%H%M")

        # 定义稳定 schema（features 存 JSON 字符串，避免 schema 演进复杂度）
        self._schema = pa.schema([
            ("inst", pa.string()),
            ("tf", pa.string()),
            ("ts_close", pa.int64()),
            ("feature_version", pa.string()),
            ("engine_id", pa.string()),
            ("features_json", pa.string()),
        ])

        self.stream_name = stream_name
        self.total_write_cnt = 0

    def _path_for(self, inst: str, tf: str) -> Path:
        # <BASE>/<inst>/<tf>/data-<runid>.parquet
        p = self.base_dir / inst / tf
        p.mkdir(parents=True, exist_ok=True)
        return p / f"data-{self._runid}.parquet"

    def _ensure_writer(self, key: Tuple[str, str]):
        if key in self._writers:
            return
        inst, tf = key
        path = self._path_for(inst, tf)
        self._files[key] = path
        # 新文件：用固定 schema 创建 writer
        writer = pq.ParquetWriter(path, self._schema, compression="zstd")  # 压缩节省空间
        self._writers[key] = writer
        self._schemas[key] = self._schema

    def _row_from_payload(self, p: Dict) -> Dict:
        return {
            "inst": p.get("inst") or "",
            "tf": p.get("tf") or "",
            "ts_close": int(p.get("ts_close") or p.get("ts") or 0),
            "feature_version": (p.get("feature_version") or ""),
            "engine_id": (p.get("engine_id") or ""),
            "features_json": json.dumps(p.get("features") or {}, ensure_ascii=False),
        }

    async def ingest(self, payload: Dict):
        """供 on_message 调用：只负责入缓冲 & 条件触发 flush（同步写）"""
        row = self._row_from_payload(payload)
        key = (row["inst"], row["tf"])
        if not key[0] or not key[1]:
            # 缺少 inst/tf 则忽略
            return
        buf = self._buffers.setdefault(key, [])
        buf.append(row)

        now = time.monotonic()
        if len(buf) >= self.flush_rows or (now - self._last_flush_mono) >= self.flush_sec:
            to_be_fluush_len = len(buf)
            await self.flush()
            logger.info(f"Stream {self.stream_name}: {to_be_fluush_len} flushed, total flushed: {self.total_write_cnt}")

    async def flush(self):
        """将所有分桶的缓冲写入到各自的 parquet（同步写；若担心阻塞可改 asyncio.to_thread）"""
        if not any(self._buffers.values()):
            self._last_flush_mono = time.monotonic()
            return

        write_cnt = 0
        for key, rows in list(self._buffers.items()):
            if not rows:
                continue
            rows_len = len(rows)
            self._ensure_writer(key)
            table = pa.Table.from_pylist(rows, schema=self._schemas[key])
            self._writers[key].write_table(table)
            # 清空缓冲
            rows.clear()
            write_cnt += rows_len
            
        self._last_flush_mono = time.monotonic()
        self.total_write_cnt += write_cnt

    async def close(self):
        """退出前 flush & 关闭所有 writer"""
        with contextlib.suppress(Exception):
            await self.flush()
        for w in self._writers.values():
            with contextlib.suppress(Exception):
                w.close()


class CSVSink:
    """
    以 (inst, tf) 为粒度写入：
      <BASE_DIR>/<inst>/<tf>/data-<runid>.csv
    - 仅使用 Python 标准库 csv
    - 采用行级 append，header 首次创建时写入
    - flush 时执行 f.flush() + os.fsync()，以降低异常退出导致的数据丢失
    """
    # FIELDS = ["inst", "tf", "ts_close", "feature_version", "engine_id", "features_json"]

    def __init__(self, base_dir: Path, flush_rows: int = 500, flush_sec: float = 2.0, stream_name: str = None):
        self.base_dir = base_dir
        self.flush_rows = flush_rows
        self.flush_sec = flush_sec

        self._buffers: Dict[Tuple[str, str], List[Dict]] = {}
        self._files: Dict[Tuple[str, str], Path] = {}
        self._fps: Dict[Tuple[str, str], object] = {}
        self._writers: Dict[Tuple[str, str], csv.DictWriter] = {}
        self._headers: Dict[Tuple[str, str], List[str]] = {}
        self._has_header_written: Dict[Tuple[str, str], bool] = {}
        self._last_flush_mono = time.monotonic()
        self._runid = datetime.now().strftime("%y%m%d_%H%M")
        self.stream_name = stream_name
        self.total_write_cnt = 0

    def _path_for(self, inst: str, tf: str) -> Path:
        p = self.base_dir / inst / tf
        p.mkdir(parents=True, exist_ok=True)
        return p / f"data-{self._runid}.csv"

    def _ensure_writer(self, key: Tuple[str, str], header: List[str]):
        if key in self._writers:
            return
        inst, tf = key
        path = self._path_for(inst, tf)
        self._files[key] = path

        # newline='' 防止多余空行；buffering=1 行缓冲；encoding=utf-8
        # 注意：部分系统上 “行缓冲” 仅对交互式终端完全生效，但我们仍然会在 flush 中强制 fsync。
        fp = open(path, mode="a", encoding="utf-8", newline="", buffering=1)
        self._fps[key] = fp
        writer = csv.DictWriter(fp, fieldnames=header, extrasaction="ignore")
        self._writers[key] = writer

        # 若是新建文件（大小为0），写 header
        has_header = path.exists() and path.stat().st_size > 0
        if not has_header:
            writer.writeheader()
            fp.flush()
            os.fsync(fp.fileno())
        self._has_header_written[key] = True

    def _row_from_payload(self, p: Dict) -> Dict:
        """
        仅输出 features_json 的键值（去除 instId / tf），不再写入 inst/tf/engine_id/feature_version 等字段。
        """
        feats = dict(p.get("features") or {})
        for k in ("instId", "tf"):
            if k in feats:
                feats.pop(k, None)
        return feats
    
    async def ingest(self, payload: Dict):
        inst = (payload.get("inst") or "").strip()
        tf   = (payload.get("tf") or "").strip()
        if not inst or not tf:
            return
        key = (inst, tf)

        row = self._row_from_payload(payload)

        buf = self._buffers.setdefault(key, [])
        buf.append(row)

        if key not in self._headers:
            header = list(row.keys())
            self._headers[key] = header
            self._ensure_writer(key, header)

        now = time.monotonic()
        if len(buf) >= self.flush_rows or (now - self._last_flush_mono) >= self.flush_sec:
            to_be_flush_len = len(buf)
            await self.flush()
            logger.info(f"Stream {self.stream_name}: {to_be_flush_len} flushed, total flushed: {self.total_write_cnt}")

    async def flush(self):
        any_buf = any(self._buffers.values())
        if not any_buf:
            self._last_flush_mono = time.monotonic()
            return

        write_cnt = 0
        for key, rows in list(self._buffers.items()):
            if not rows:
                continue
            header = self._headers.get(key)
            if not header:
                # 若出现异常情况（无 header），用首条数据的键补救
                header = sorted(list(rows[0].keys()))
                self._headers[key] = header
                self._ensure_writer(key, header)

            writer = self._writers[key]
            fp = self._fps[key]
            for r in rows:
                # 按 header 顺序写入，缺失补空，多余键被忽略（extrasaction="ignore"）
                writer.writerow({h: r.get(h, "") for h in header})
            rows_len = len(rows)
            rows.clear()
            fp.flush()
            os.fsync(fp.fileno())
            write_cnt += rows_len

        self._last_flush_mono = time.monotonic()
        self.total_write_cnt += write_cnt

    async def close(self):
        with contextlib.suppress(Exception):
            await self.flush()
        # 关闭所有文件句柄
        for fp in self._fps.values():
            with contextlib.suppress(Exception):
                fp.flush()
                os.fsync(fp.fileno())
                fp.close()
                
# ========== 主消费程序 ==========
async def main(args):
    # 以 CLI 参数为准（未提供则由 build_parser 的 ENV 默认兜底）
    base_dir    = Path(args.base_dir)
    redis_dsn   = args.redis_dsn
    start_pos   = args.start
    concurrency = args.concurrency

    streams: List[str] = []
    if args.streams:
        streams = args.streams
    else:
        env_multi = os.getenv("REDIS_STREAMS")
        if env_multi:
            streams = [s.strip() for s in env_multi.split(",") if s.strip()]
        elif args.stream or os.getenv("REDIS_STREAM"):
            streams = [args.stream or os.getenv("REDIS_STREAM")]
        else:
            streams = [STREAM_NAME]

    sinks: Dict[str, ParquetSink] = {}
    subs: Dict[str, RedisStreamsSubscriber] = {}
    tasks: List[asyncio.Task] = []


    for s in streams:
        # sink = ParquetSink(base_dir, flush_rows=FLUSH_ROWS, flush_sec=FLUSH_SEC, stream_name=s)
        sink = CSVSink(base_dir, flush_rows=FLUSH_ROWS, flush_sec=FLUSH_SEC, stream_name=s)
        
        sinks[s] = sink

        sub = RedisStreamsSubscriber(
            dsn=redis_dsn,
            stream=s,
            start=start_pos,
            block_ms=3000,
            fetch_count=256,
            concurrency=concurrency,
        )
        subs[s] = sub

        async def _make_on_message(stream_name: str):
            async def on_message(payload: Dict):
                # if payload.get("inst") and payload["inst"] != _inst_from_stream(stream_name):
                #     logger.warning(f"inst mismatch: stream={stream_name}, payload.inst={payload.get('inst')}")
                await sinks[stream_name].ingest(payload)
            return on_message

        on_msg = await _make_on_message(s)
        # logger.info(f"[parquet] tailing '{s}' from {START_POS}, base={BASE_DIR}")
        logger.info(f"[csv] tailing '{s}' from {START_POS}, base={BASE_DIR}")
        tasks.append(asyncio.create_task(sub.run_forever(on_msg), name=f"subscriber:{s}"))

    stop_event = asyncio.Event()
    def _graceful(*_):
        stop_event.set()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, _graceful)

    await stop_event.wait()

    for t in tasks:
        t.cancel()
    for t in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await t

    for sink in sinks.values():
        await sink.close()


if __name__ == "__main__":
    args = build_parser().parse_args()
    os.environ.setdefault("REDIS_DSN", args.redis_dsn)
    if args.streams:
        os.environ.setdefault("REDIS_STREAMS", ",".join(args.streams))
    if args.stream:
        os.environ.setdefault("REDIS_STREAM", args.stream)
    os.environ.setdefault("BASE_DIR", args.base_dir)
    os.environ.setdefault("START", args.start)
    os.environ.setdefault("FLUSH_ROWS", str(args.flush_rows))
    os.environ.setdefault("FLUSH_SEC", str(args.flush_sec))
    os.environ.setdefault("CONCURRENCY", str(args.concurrency))
    asyncio.run(main(args))