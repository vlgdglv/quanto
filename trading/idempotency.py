# trading/idempotency.py
import time, uuid

def now_ms() -> int:
    """Monotonic-ish current time in milliseconds (for expTime, timestamps)."""
    return int(time.time() * 1000)

def make_cl_ord_id(prefix: str = "algo") -> str:
    """Generate a client order id for idempotency and tracking."""
    return f"{prefix}-{uuid.uuid4().hex[:16]}"
