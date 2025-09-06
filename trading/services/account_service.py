# trading/services/account_service.py
import re
import logging
from typing import List, Optional
from trading.models import Position, Balance
from trading.enums import TdMode


def _to_float_or_none(x) -> Optional[float]:
    # OKX 很多字段可能返回 ""，这里统一转 None；合法数字字符串转 float
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    x = str(x).strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None

def _to_float_zero_if_empty(x) -> float:
    # 对于像 pos / availPos 这类数量，空串按 0.0 处理更顺手
    v = _to_float_or_none(x)
    return v if v is not None else 0.0

class AccountService:
    """
    Account/position/leverage/mode configuration and queries.
    """

    def __init__(self, http_client, endpoints) -> None:
        self._http = http_client
        self._ep = endpoints
        self.log = getattr(http_client, "log", logging.getLogger("AccountService"))


    async def get_config(self) -> dict:
        """Return account config (position mode, Greeks, etc.)."""
        ...

    async def set_position_mode(self, net: bool) -> None:
        """Set position mode to net or long/short; requires no open pos/orders."""
        ...

    async def set_leverage(self, instId: str, lever: int, mgnMode: TdMode) -> None:
        """Set leverage for an instrument and margin mode."""
        ...

    async def get_max_avail_size(self, instId: str, mgnMode: TdMode, ccy: Optional[str] = None) -> float:
        """Return maximum available size for order placement."""
        ...

    async def get_positions(self, 
                            instType:  Optional[str] = "SWAP", 
                            instId: Optional[str] = None) -> List[Position]:
        """
        查询当前持仓（默认仅 SWAP 永续）。
        与 OKX 一致支持：
          - instType 可为 None（不传）
          - instId 可选，支持精确过滤
        冲突策略：
          - 若传入 instType 与由 instId 推断的类型不一致，则以 instId 推断为准，
            自动覆盖 instType，并输出 warning 日志。
        """
        path = getattr(self._ep, "account_positions", None) or \
               (self._ep.get("account_positions") if isinstance(self._ep, dict) else None) or \
               "/api/v5/account/positions"

        # -------- 参数与冲突处理 --------
        params: dict = {}
        # 如果给了 instId，尝试从 instId 推断类型
        inferred_type = self._inst_type_from_inst_id(instId) if instId else None

        # 规范化大小写（OKX 使用大写）
        instType_norm = instType.upper() if isinstance(instType, str) else None
        inferred_norm = inferred_type.upper() if isinstance(inferred_type, str) else None

        # 若两者都有且冲突：以 instId 推断为准，覆盖 instType
        if instType_norm and inferred_norm and instType_norm != inferred_norm:
            self.log.warning(
                "get_positions: instType (%s) conflicts with inferred from instId (%s). "
                "Using instId-inferred type: %s",
                instType_norm, instId, inferred_norm
            )
            instType_norm = inferred_norm

        # 构建请求参数（与 OKX 接口一致）
        if instType_norm:
            params["instType"] = instType_norm
        if instId:
            params["instId"] = instId

        # -------- 发起请求 --------
        resp = await self._http.get_private(path, params=params)
        raw_list = resp.get("data", []) or []

        # -------- 结果本地再保险过滤（防御性）--------
        # 如果 instType 最终有效，确保只保留该类型
        # 如果 instId 指定，确保只保留该 instId
        positions: List[Position] = []
        for it in raw_list:
            if instType_norm and it.get("instType") != instType_norm:
                continue
            if instId and it.get("instId") != instId:
                continue

            positions.append(
                Position(
                    # --- 基本识别 ---
                    instType=it.get("instType", ""),
                    instId=it.get("instId", ""),
                    posId=it.get("posId", ""),
                    posSide=(it.get("posSide") or "net"),
                    mgnMode=(it.get("mgnMode") or ""),

                    # --- 数量与价格 ---
                    pos=_to_float_zero_if_empty(it.get("pos")),
                    availPos=_to_float_zero_if_empty(it.get("availPos")),
                    avgPx=_to_float_or_none(it.get("avgPx")),
                    markPx=_to_float_or_none(it.get("markPx")),
                    liqPx=_to_float_or_none(it.get("liqPx")),
                    lever=_to_float_or_none(it.get("lever")),

                    # --- 盈亏与名义价值 ---
                    upl=_to_float_or_none(it.get("upl")),
                    uplRatio=_to_float_or_none(it.get("uplRatio")),
                    notionalUsd=_to_float_or_none(it.get("notionalUsd")),

                    # --- 保证金相关 ---
                    imr=_to_float_or_none(it.get("imr")),
                    mmr=_to_float_or_none(it.get("mmr")),
                    margin=_to_float_or_none(it.get("margin")),
                    mgnRatio=_to_float_or_none(it.get("mgnRatio")),

                    # --- 其他 ---
                    adl=int(it.get("adl")) if str(it.get("adl") or "").isdigit() else None,
                    cTime=int(it.get("cTime")) if str(it.get("cTime") or "").isdigit() else None,
                    uTime=int(it.get("uTime")) if str(it.get("uTime") or "").isdigit() else None,
                )
            )
        return positions

    async def get_balance(self, ccy: str = "USDT") -> Balance:
        """Return balance object for given currency."""
        ...

    @staticmethod
    def _inst_type_from_inst_id(instId: str) -> Optional[str]:
        """
        根据 OKX instId 推断交易品种类型。

        - BTC-USDT-SWAP       -> SWAP
        - BTC-USDT            -> SPOT
        - BTC-USD-230915      -> FUTURES
        - BTC-USD-230915-40000-C -> OPTION
        """
        if not instId or "-" not in instId:
            return None

        parts = instId.split("-")
        last = parts[-1].upper()

        if len(parts) == 2:
            return "SPOT"
        if last == "SWAP":
            return "SWAP"
        if re.fullmatch(r"\d{6}", last):
            return "FUTURES"
        if len(parts) >= 5 and last in ("C", "P"):
            return "OPTION"

        # other cases
        return None