# utils/time.py
from datetime import datetime, timezone, timedelta

def utc_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

def parse_tf(tf: str) -> int:
    if tf.endswith("ms"):
        return int(tf[:-2])
    if tf.endswith("s"):
        return int(tf[:-1]) * 1000
    if tf.endswith("m"):
        return int(tf[:-1]) * 60_000
    if tf.endswith("h"):
        return int(tf[:-1]) * 3_600_000
    if tf.endswith("d"):
        return int(tf[:-1]) * 86_400_000
    raise ValueError(f"unknown timeframe: {tf}")

def floor_bucket(ts_ms: int, tf_ms: int) -> int:
    return ts_ms - (ts_ms % tf_ms)

def bucket_right(ts_ms: int, tf_ms: int) -> int:
    left = floor_bucket(ts_ms, tf_ms)
    return left + tf_ms
