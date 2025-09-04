# trading/config.py
from dataclasses import dataclass
from trading.enums import Env

@dataclass
class TradingSettings:
    """Trading runtime configuration."""
    env: Env
    api_key: str
    api_secret: str
    passphrase: str

    simulated_header: bool = True       # add x-simulated-trading: 1 when env=paper
    recv_window_ms: int = 5000
    http_timeout_s: int = 5
    rate_limit_rps: float = 8.0

    default_td_mode: str = "cross"      # "cross" | "isolated"
    default_pos_mode_net: bool = True   # True=net position mode, False=long/short
