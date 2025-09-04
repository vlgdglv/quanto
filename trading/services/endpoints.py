# trading/services/endpoints.py
from dataclasses import dataclass
from trading.enums import Env

@dataclass
class Endpoints:
    rest_base: str
    ws_public: str
    ws_private: str
    use_simulated_header: bool

def make_endpoints(env: Env, simulated_header: bool = True) -> Endpoints:
    """
    Single-domain strategy; switch paper/live via header.
    """
    return Endpoints(
        rest_base="https://www.okx.com",
        ws_public="wss://ws.okx.com:8443/ws/v5/public",
        ws_private="wss://ws.okx.com:8443/ws/v5/private",
        use_simulated_header=(env == Env.PAPER and simulated_header),
    )
