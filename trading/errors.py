# trading/errors.py
class TradingError(Exception):
    """Base trading error."""

class PrecisionError(TradingError):
    """Price/size precision or min/max constraints violated."""

class MarginError(TradingError):
    """Insufficient margin or size exceeds max available."""

class ReduceOnlyError(TradingError):
    """Reduce-only rule violated or would increase exposure."""

class SlippageError(TradingError):
    """Requested price violates max slippage constraint."""

class ReconcileTimeout(TradingError):
    """Timed out while waiting for order reconciliation/ack."""

class OkxApiError(TradingError):
    """Wrapper for OKX API error codes/messages."""
    def __init__(self, code: str, msg: str):
        super().__init__(f"OKX[{code}]: {msg}")
        self.code = code
        self.msg = msg
