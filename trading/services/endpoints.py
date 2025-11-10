# trading/services/endpoints.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Endpoints:
    # 主机域名/基址
    rest_base: str
    ws_public: str
    ws_private: str
    ws_business: Optional[str]
    # 是否在请求头加 x-simulated-trading: 1（仅 paper）
    use_simulated_header: bool

    # 常用 REST 路径常量（便于统一引用）
    public_instruments: str = "/api/v5/public/instruments"
    trade_order: str = "/api/v5/trade/order"
    trade_cancel: str = "/api/v5/trade/cancel-order"
    trade_amend: str = "/api/v5/trade/amend-order"
    trade_order_query: str = "/api/v5/trade/order"
    trade_orders_pending: str = "/api/v5/trade/orders-pending"
    trade_fills_history: str = "/api/v5/trade/fills-history"
    account_positions: str = "/api/v5/account/positions"
    account_balance: str = "/api/v5/account/balance"


# trading/services/endpoints.py (续)
def make_endpoints_from_cfg(cfg: dict) -> Endpoints:
    """
    依据 config.yaml 选择 LIVE/PAPER 的 REST/WS 域名，
    并根据 trading.simulated_header & env 是否 PAPER 决定是否启用模拟头。
    """
    try:
        env_str = str(cfg["trading"]["env"]).upper()   # "PAPER" | "LIVE"
        env_key = "paper" if env_str == "PAPER" else "live"

        rest_base = cfg["okx"]["rest_base"][env_key].rstrip("/")

        ws_cfg = cfg["okx"]["ws"][env_key]
        ws_public = ws_cfg["public"]
        ws_private = ws_cfg["private"]
        ws_business = ws_cfg.get("business")  # live 下可能也有，paper 通常必备

        # x-simulated-trading: 仅 PAPER 且配置允许时才启用
        use_sim = (env_str == "PAPER") and bool(cfg["trading"].get("simulated_header", True))

    except KeyError as e:
        raise ValueError(f"Invalid cfg missing key: {e}") from e

    return Endpoints(
        rest_base=rest_base,
        ws_public=ws_public,
        ws_private=ws_private,
        ws_business=ws_business,
        use_simulated_header=use_sim,
    )
