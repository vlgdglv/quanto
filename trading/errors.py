# trading/errors.py
class TradingError(Exception):
    """Base trading error."""
    def __init__(self, msg: str = ""):
        super().__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg or self.__class__.__name__

class PrecisionError(TradingError):
    """Price/size precision or min/max constraints violated."""

    def __init__(self, msg: str = "", **ctx):
        super().__init__(msg)
        self.ctx = ctx

    def __str__(self):
        base = super().__str__()
        if self.ctx:
            details = ", ".join(f"{k}={v}" for k, v in self.ctx.items())
            return f"{base} [{details}]"
        return base
    
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
