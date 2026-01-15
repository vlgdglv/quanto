# llm_agents.py
import re, math

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional, Tuple, Dict, Any, Iterable, Mapping

from agent.agent_hub.llm_factory import get_chat_model, cb
from agent.states import RDState
from agent.schemas import FeatureFrame
from utils.logger import logger

Side = Literal["LONG","SHORT","FLAT"]


def _is_ok(v: Any) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))

def _compact(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if _is_ok(v)}

def _match_any(name: str, pats: Iterable[str]) -> bool:
    for p in pats:
        pat = "^" + re.escape(p).replace(r"\*", ".*") + "$"
        if re.match(pat, name):
            return True
    return False

def filter_features(
    feats: Dict[str, Any],
    include: Iterable[str],
    exclude: Optional[Iterable[str]] = None
) -> Dict[str, Any]:
    exclude = exclude or []
    out: Dict[str, Any] = {}
    for k, v in feats.items():
        if _match_any(k, include) and not _match_any(k, exclude) and _is_ok(v):
            out[k] = v
    return out

def _filter_feats(feats: Mapping[str, Any], 
                  include: Iterable[str], 
                  exclude: Iterable[str] = ()
                  ) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in feats.items():
        if _match_any(k, include) and not _match_any(k, exclude):
            out[k] = v
    return out


def build_agent_snapshot(frame: FeatureFrame) -> FeatureFrame:
    # print("Custom snapshot for tf:", tf)
    tf = frame.tf
    if tf in ("1H", "4H"):
        return build_rd_snapshot(frame)
    if tf in ("15m", ):
        return build_timing_snapshot(frame)
    logger.warning(f"nothing to do for tf")
    return frame


# ======================== RD Agent ========================

_RD_INC = [
    # 趋势/区间
    "ema_fast", "ema_slow", "macd_dif", "macd_dea", "macd_hist",
    "rsi", "kdj_k", "kdj_d", "kdj_j",  # ✅ 加回 KDJ（作为动量/震荡佐证）
    "er",
    "donchian_width", "donchian_width_norm",
    "s_donchian_dist_upper", "s_donchian_dist_lower",
    # 量能/参与度
    "cvd", "oi", "d_oi", "d_oi_rate", "oi_ema",
    # 波动
    "atr", "rv_ewma", "vol_ratio_1h_to_4h",
    # 资金
    "funding_premium_z", "funding_time_to_next_min",
    # 多尺度稳定性（方向一致性）
    "s_mom_slope_H60m", "s_mom_slope_H180m", "s_mom_slope_H420m",
    "s_cvd_delta_H60m", "s_cvd_delta_H180m", "s_cvd_delta_H420m",
    "s_oi_rate_H60m", "s_oi_rate_H180m", "s_oi_rate_H420m",
    # 上层汇总（若存在）
    "trend_agreement", "conflict_score", "basis_zscore", "perp_index_basis_z*",
]

_RD_EXC = [
    # 噪声/微结构：明确剔除
    "spread_bp", "ofi_5s", "microprice", "kyle_lambda", "vpin",
    "qi1", "qi5",
    # 多尺度里不需要的微结构均值/流动性
    "s_spread_bp_mean_*", "s_kyle_ema_*", "s_vpin_mean_*",
    # 非主干的多类 Donchian 距离（保留 s_donchian_dist_*）
    "donchian_dist_*",
]

def build_rd_snapshot(frame: FeatureFrame) -> FeatureFrame:
    feats: Dict[str, Any] = frame.features

    # 先总体筛一遍
    f = _filter_feats(feats, _RD_INC, _RD_EXC)

    # 再分组（字段都来自已筛集合 f，保证精简）
    trend = _filter_feats(f, ["ema_*", "macd_*", "rsi", "kdj_*", "er", "s_mom_slope_*"])
    regime = _filter_feats(f, ["donchian_width*", "s_donchian_dist_*", "rv_ewma", "atr", "vol_ratio_1h_to_4h"])
    participation = _filter_feats(f, ["cvd", "oi", "d_oi", "d_oi_rate", "oi_ema", "s_oi_rate_*", "s_cvd_delta_*"])
    funding = _filter_feats(f, ["funding_premium_z", "funding_time_to_next_min"])
    summary = _filter_feats(f, ["trend_agreement", "conflict_score", "basis_zscore", "perp_index_basis_z*"])

    return FeatureFrame(
        inst=frame.inst,
        tf=frame.tf,
        ts_close=frame.ts_close,
        features={
            "trend": trend,
            "regime": regime,
            "participation": participation,
            "funding": funding,
            "summary": summary,
        },
        kind="RD frame",
    )
     

# ================================================  
#
#   Regime&Direction Agent  
#
# ================================================  

class RDOut(BaseModel):
    regime: Side
    regime_confidence: float
    direction: Side
    direction_confidence: float
    reasons: List[str] = []
    invalidation: List[str] = []
    
    @field_validator("reasons", "invalidation", mode="before")
    @classmethod
    def _coerce_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v.strip()]
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if x is not None]
        return []


_rd_parser = PydanticOutputParser(pydantic_object=RDOut)
_format_instructions = _rd_parser.get_format_instructions()


_rd_prompt = ChatPromptTemplate.from_template("""
You are a market regime & direction analyst for {inst}. 
You bear a critical responsibility — to deliver highly accurate signals for a system targeting 3–7%% daily returns.
Use ONLY these feature snapshots:

H4={snap4h}
H1={snap1h}

Think like a discretionary trader, not a rule engine.
Infer the underlying market state from relationships between features
(EMA, MACD, RSI, Donchian, ATR, CVD, OI, funding, etc.).
You may assume the data is internally consistent and scaled.

Guidelines:
- H4 gives the structural regime; H1 refines short-term bias or momentum shifts.
- Prefer giving a directional view (LONG/SHORT) if the evidence leans clearly one way.
- Choose FLAT only when both frames show no clear momentum or conflicting signals.
- Use participation (CVD, OI, d_oi_rate) and volatility (ATR, width_norm) as conviction checks.
- Funding and crowding signals should only temper confidence, not dominate it.
- Keep reasoning compact and intuitive; no math or thresholds.

Output JSON ONLY (no prose, no code fences), with the following fields:
{{
 "regime": "LONG|SHORT|FLAT",
 "regime_confidence": 0.0,
 "direction": "LONG|SHORT|FLAT",
 "direction_confidence": 0.0,
 "reasons": ["<=3 concise intuitive explanations"],
 "invalidations": ["1–3 brief invalidation cues or regime breaks"]
}}
""")

_rd_chain = (
    _rd_prompt.partial(format_instructions=_format_instructions)
    | get_chat_model(task="rd")
    | _rd_parser
)

async def run_rd_agent(inst, snap4h, snap1h) -> RDOut:
    raw = await _rd_chain.ainvoke({
        "inst": inst,
        "snap4h": snap4h.model_dump(),
        "snap1h": snap1h.model_dump(),
    })
    if isinstance(raw, dict):
        inv = raw.get("invalidation")
        if isinstance(inv, str):
            raw["invalidation"] = [inv]
        return RDOut(**raw)
    if isinstance(raw.invalidation, str):
        raw = raw.copy(update={"invalidation": [raw.invalidation]})
    return raw

# ================================================  
#
#   Regime Agent  
#
# ================================================  
class RegimeOut(BaseModel):
    regime: Side
    regime_confidence: float = Field(ge=0.0, le=1.0)
    reasons: List[str] = []
    invalidation: List[str] = []

_regime_prompt = ChatPromptTemplate.from_template("""
You are the STRUCTURAL regime analyst for {inst}.
Deliver highly accurate signals for a system targeting 3–7%% daily returns.
Use the 4H snapshot:
H4={snap4h}
Think like a discretionary trader, not a rule engine.
Infer regime from relationships among EMA/MACD/RSI, Donchian/ATR (width/expansion),
participation (CVD/OI/d_oi_rate), and funding/crowding only as confidence tempering.
Keep reasoning compact and intuitive; no math or thresholds.
Guidelines:
- Output the structural regime: LONG / SHORT / FLAT.
- Prefer LONG/SHORT if evidence is clear; choose FLAT only when signal is mixed.
- Participation & volatility act as conviction checks; funding only tempers confidence.
- JSON ONLY (no prose, no code fences).
Return:
{{
 "regime": "LONG|SHORT|FLAT",
 "regime_confidence": 0.0,
 "reasons": ["<=3 concise intuitive explanations"],
 "invalidation": ["1–3 brief regime-break cues"]
}}
""")

_regime_parser = PydanticOutputParser(pydantic_object=RegimeOut)
_regime_format_instructions = _regime_parser.get_format_instructions()

_regime_chain = (
    _regime_prompt.partial(format_instructions=_regime_format_instructions)
    | get_chat_model(task="rd")
    | _regime_parser
)

async def run_regime_agent(inst: str, snap4h) -> RegimeOut:
    raw = await _regime_chain.ainvoke({
        "inst": inst,
        "snap4h": snap4h.model_dump(),   # 只传 H4
    })
    
    if isinstance(raw, dict):
        inv = raw.get("invalidation")
        if isinstance(inv, str):
            raw["invalidation"] = [inv]
        return RegimeOut(**raw)
    
    data = getattr(raw, "dict", lambda: getattr(raw, "model_dump", lambda: {}) )()
    if isinstance(data, dict):
        if isinstance(data.get("invalidation"), str):
            data["invalidation"] = [data["invalidation"]]
        return RegimeOut(**data)
    raise ValueError("Regime agent returned unexpected payload")

# ================================================  
#
#   Direction Agent  
#
# ================================================  
class DirectionOut(BaseModel):
    direction: Side
    direction_confidence: float = Field(ge=0.0, le=1.0)
    reasons: List[str] = []
    invalidation: List[str] = []

_direction_prompt = ChatPromptTemplate.from_template("""
You are the TACTICAL direction analyst for {inst}.
Deliver highly accurate short-term bias for a system targeting 3–7%% daily returns.
Use the 1H snapshot:
H1={snap1h}
Think like a discretionary trader, not a rule engine.
Focus on marginal shifts in EMA/MACD/RSI, short-horizon momentum slope,
Donchian context and ATR, with participation (CVD/OI/d_oi_rate) as confirmation.
Funding/crowding only tempers confidence. Keep reasoning compact.
Guidelines:
- Output tactical bias: LONG / SHORT / FLAT.
- Prefer LONG/SHORT if evidence leans clearly; choose FLAT only when mixed/conflicting.
- Participation & volatility modulate conviction; funding only tempers confidence.
- JSON ONLY (no prose, no code fences).
Return:
{{
 "direction": "LONG|SHORT|FLAT",
 "direction_confidence": 0.0,
 "reasons": ["<=3 concise intuitive explanations"],
 "invalidation": ["1–3 brief reversal/weakening cues"]
}}
""")


_direction_parser = PydanticOutputParser(pydantic_object=DirectionOut)
_direction_format_instructions = _direction_parser.get_format_instructions()

_direction_chain = (
    _direction_prompt.partial(format_instructions=_direction_format_instructions)
    | get_chat_model(task="rd")
    | _direction_parser
)

async def run_direction_agent(inst: str, snap1h) -> DirectionOut:
    raw = await _direction_chain.ainvoke({
        "inst": inst,
        "snap1h": snap1h.model_dump(),
    })
    if isinstance(raw, dict):
        inv = raw.get("invalidation")
        if isinstance(inv, str):
            raw["invalidation"] = [inv]
        return DirectionOut(**raw)
    data = getattr(raw, "dict", lambda: getattr(raw, "model_dump", lambda: {}) )()
    if isinstance(data, dict):
        if isinstance(data.get("invalidation"), str):
            data["invalidation"] = [data["invalidation"]]
        return DirectionOut(**data)
    raise ValueError("Direction agent returned unexpected payload")



# ================================================  
#
#   Timing Agent  
#
# ================================================  
_TM_INC = [
    # 上层方向约束（可能来自上游，不一定在表里）
    "ema_fast", "ema_slow", "macd_dif", "macd_dea", "macd_hist", "rsi", "er",
    # 微结构（真实列）
    "spread_bp", "ofi_5s", "microprice",
    # 近窗波动（15m 上用 atr/rv_ewma 直接代表）
    "atr_15m", "atr", "rv_ewma", "realized_vol_5m", "realized_vol_15m", "jump_indicator",
    # 资金门槛（真实列）
    "funding_premium_z", "funding_time_to_next_min",
    # 区间定位（真实列）
    "donchian_upper", "donchian_lower", "donchian_width_norm", "s_donchian_dist_upper", "s_donchian_dist_lower",
]
_TM_EXC = [
    # 1H/4H 的冗余在 15m 弱化
    "cvd", "oi", "d_oi", "d_oi_rate", "oi_ema",
    "s_mom_slope_*", "s_cvd_delta_*", "s_oi_rate_*",
    "kyle_lambda", "vpin", "qi1", "qi5",
]

def build_timing_snapshot(frame: FeatureFrame) -> FeatureFrame:
    feats: Dict[str, Any] = frame.features

    f = _filter_feats(feats, _TM_INC, _TM_EXC)

    trend = _filter_feats(f, ["ema_*", "macd_*", "rsi", "kdj_*", "er",])
    # higher = _filter_feats(f, ["direction", "confidence", "regime", "trend_agreement"])
    micro  = _filter_feats(f, ["spread_bp", "ofi_5s", "microprice"])
    vol    = _filter_feats(f, ["atr_15m", "atr", "rv_ewma", "realized_vol_5m", "realized_vol_15m", "jump_indicator"])
    zone   = _filter_feats(f, ["donchian_upper", "donchian_lower", "donchian_width_norm", "s_donchian_dist_*"])
    funding= _filter_feats(f, ["funding_premium_z", "funding_time_to_next_min"])
    
    return FeatureFrame(
        inst=frame.inst,
        tf=frame.tf,
        ts_close=frame.ts_close,
        features= {
            # "higher": higher,   # 顺势/一致性
            "small_trend": trend,
            "micro": micro,     # 微结构节拍
            "vol": vol,         # 近窗风险
            "zone": zone,       # 入场/止盈区间
            "funding": funding  # 资金门槛
        },
        kind="TIMING frame",
    )

class TimingOut(BaseModel):
    action: Literal["BUY", "SELL", "HOLD", "CLOSE", "SKIP"]
    confidence: float
    side: Literal["LONG", "SHORT", "NONE"]
    reasons: List[str] = []
    invalidation: List[str] = []

    @field_validator("reasons", "invalidation", mode="before")
    @classmethod
    def _coerce_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v.strip()]
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if x is not None]
        return []

    @field_validator("confidence")
    @classmethod
    def _clip_confidence(cls, v):
        try:
            v = float(v)
        except Exception:
            v = 0.0
        return max(0.0, min(1.0, v)) 

# _timing_prompt = ChatPromptTemplate.from_template("""
# You are a timing agent for {inst}. Higher timeframe context:
# 4 hour: regime={regime}({regime_conf}); 1 hour: direction={direction}({dir_conf})

# Use the 15m snapshot to produce ONE decisive plan.

# STRICT RULES:
# - Output JSON ONLY (no code fences, no extra text).
# - The field `action` MUST be ONE of: "BUY", "SELL", "ADD", "REDUCE", "SKIP".
#   * "BUY" = go long; "SELL" = go short.
#   * If microstructure conflicts with the higher timeframe, use "SKIP".
# - Also return: entry_zone [low, high], stop, 1–3 tps (numbers), and short notes (array of strings).
# - Keep entry_zone narrow; avoid over-trading/cycling decisions.

# 15m: {snap15}

# {format_instructions}
# """)

# _timing_prompt = ChatPromptTemplate.from_template("""
# You are a decisive 15m timing agent for {inst}.
# Aim to act; skip only on hard disqualifiers.

# Higher TF (context, must be considered):
# 4 hour: regime={regime}({regime_conf}); 1 hour: direction={direction}({dir_conf})

# Position: {pos_info}   # 'NONE' if flat

# 15m snapshot (read-only): {snap15}

# Policy (concise, aggressive):
# - Prefer action in the HTF direction. Overbought/oversold ≠ reversal by itself.
# - Micro (spread_bp, ofi_5s, microprice) drives timing; zone (donchian_*, width_norm) provides context only.
# - Funding gate: if |funding_premium_z|>1.5 against the planned side OR time_to_next_min<120s → strongly consider SKIP/REDUCE.
# - Allowed actions: BUY, SELL, ADD, REDUCE, SKIP.
#   • ADD/REDUCE only if a same-side position exists; if flat, use BUY/SELL (not ADD/REDUCE).
# - No numbers (no targets/stops/zones). Output must be JSON only.

# Return JSON with fixed minimal keys:
# {{
#   "action":"BUY|SELL|ADD|REDUCE|SKIP",
#   "confidence":0.0,            // one decimal, 0~1
#   "sizing":"LIGHT|NORMAL|AGGRESSIVE",
#   "reasons":["<=3 short, evidence-based"],
#   "invalidation":["<=2 brief stand-down cues"]
# }}
# """)

_timing_prompt = ChatPromptTemplate.from_template("""
You are a decisive 15m timing agent for {inst}.
MVP rule: FULL in/out only. No partial add/reduce. Be decisive; skip only on hard disqualifiers.
Higher TF (context, consider but don't overrule if signals are clear):
4 hour: regime={regime}({regime_conf}); 1 hour: direction={direction}({dir_conf})
Position: {pos_info}
15m snapshot (read-only): {snap15}
Action set:
- If Position==NONE → choose one: BUY (open long) | SELL (open short) | SKIP.
- If Position!=NONE → choose one: HOLD (maintain) | CLOSE (full exit) | SKIP.
Guidelines (concise and aggressive):
- Follow higher timeframe bias when consistent with microtrend; if strongly conflicting, SKIP.
- Use micro (spread_bp, ofi_5s, microprice) for timing and zone (donchian_*, width_norm) as context.
- Funding gate: if |funding_premium_z|>1.5 against the planned side OR time_to_next_min<120s → SKIP/CLOSE.
- No numeric outputs. Return JSON only.
Return JSON (fixed keys; 'side' only for BUY/SELL):
{{
  "action": "BUY|SELL|HOLD|CLOSE|SKIP",
  "side": "LONG|SHORT|NONE",
  "confidence": 0.0,
  "reasons": [
     "<=4 short, evidence-based points, including at least one based on position state (e.g., 'already long and momentum still strong', 'flat so eligible to open short')."
  ],
}}
""")

#   "invalidation": ["<=2 brief stand-down cues"]

def format_position_for_prompt(positions, inst: str) -> str:
    """
    positions: List[Position]（你的 dataclass）
    返回 'NONE' 或 'net=LONG/SHORT size avg mark liq lev uplr' 的短字符串
    """
    if not positions:
        return "NONE"
    
    def fmt_one(p):
        size = float(getattr(p, "pos", 0.0))
        if abs(size) < 1e-12:
            return None
        side = "LONG" if size > 0 else "SHORT"
        avg  = float(getattr(p, "avgPx", 0.0) or 0.0)
        mark = float(getattr(p, "markPx", 0.0) or 0.0)
        liq  = float(getattr(p, "liqPx", 0.0) or 0.0)
        lev  = float(getattr(p, "lever", 0.0) or 0.0)
        uplr = float(getattr(p, "uplRatio", 0.0) or 0.0)

        def r(x, n=6):
            return f"{x:.6f}".rstrip("0").rstrip(".")

        inst = getattr(p, "instId", "?")
        return f"{inst}: net={side} size={r(abs(size),3)} @avg={r(avg)} mark={r(mark)} liq={r(liq)} lev={r(lev,2)} uplr={uplr:.4f}"

    filtered = [p for p in positions if inst is None or getattr(p, "instId", None) == inst]
    if not filtered:
        return "No open positions."

    parts = [fmt_one(p) for p in filtered if fmt_one(p)]
    return " | ".join(parts) if parts else "NONE"


_timing_parser = PydanticOutputParser(pydantic_object=TimingOut)
_format_instructions = _timing_parser.get_format_instructions()


_timing_chain = (
    _timing_prompt.partial(format_instructions=_format_instructions)
    | get_chat_model(task="timing") 
    | _timing_parser
    )

async def run_timing_agent(inst, rd: RDState, snap15, snap30=None, positions=None) -> TimingOut:
    raw = await _timing_chain.ainvoke({
        "inst": inst,
        "regime": rd.regime, "regime_conf": rd.regime_confidence,
        "direction": rd.direction, "dir_conf": rd.direction_confidence,
        "snap15": snap15.model_dump(),
        "snap30": snap30.model_dump() if snap30 else None,
        "pos_info": format_position_for_prompt(positions, inst),
    })
    return raw

# ================================================  
#
#   Flat Timing Agent  
#
# ================================================  
flat_timing_prompt = ChatPromptTemplate.from_template("""
You are a decisive 15m FLAT agent for {inst}. (No position currently.)
Be aggressive; skip only on hard disqualifiers.
Higher TF:
4 hour: regime={regime}({regime_conf}); 1 hour: direction={direction}({dir_conf})
Position: NONE
15m snapshot (read-only): {snap15}
Policy:
- Prefer acting with HTF bias (LONG/SHORT). Micro (spread_bp, ofi_5s, microprice) drives timing; zone (donchian_*, width_norm) gives context.
- If H4 missing (regime is None) → use H1 as bias; if H1 missing  (regime is None) → use H4; if both missing → SKIP (note 'no HTF context').                                                
- Funding gate: if |funding_premium_z|>1.5 against the planned side OR time_to_next_min<120s → SKIP.
- No numeric outputs; JSON only.

Return JSON:
{{
  "action":"BUY|SELL|SKIP",
  "side":"LONG|SHORT|NONE",
  "confidence":0.0,
  "reasons":["<=3 short, include one based on being flat (e.g., 'flat so eligible to open short with HTF SHORT')"],
}}
""")
# "invalidation":["<=2 brief cues to SKIP"]

_flat_timing_chain = (
    flat_timing_prompt.partial(format_instructions=_format_instructions)
    | get_chat_model(task="timing") 
    | _timing_parser
    )

async def run_flat_timing_agent(inst, rd: RDState, snap15, snap30=None, positions=None) -> TimingOut:
    raw = await _flat_timing_chain.ainvoke({
        "inst": inst,
        "regime": rd.regime, "regime_conf": rd.regime_confidence,
        "direction": rd.direction, "dir_conf": rd.direction_confidence,
        "snap15": snap15.model_dump(),
        "snap30": snap30.model_dump() if snap30 else None,
    }, 
    # config={"callbacks": cb, "run_name": "run_rd_agent", "tags": ["rd", inst]},
    )
    return raw

# "pos_info": format_position_for_prompt(positions, inst),
# ================================================  
#
#   Position Timing Agent  
#
# ================================================  
_position_timing_prompt = ChatPromptTemplate.from_template("""
You are a decisive 15m POSITION agent for {inst}. (There is an existing position.)
Your goal: manage the current position to achieve stable daily returns between **+3% and +7%** while avoiding sharp drawdowns.

Context:
- 4H regime={regime}({regime_conf}); 1H direction={direction}({dir_conf})
  (Treat higher timeframe bias as a decaying influence; respect it, but prioritize local signals and risk state.)
- If H4 missing (regime is None) → use H1 as bias; if H1 missing  (regime is None) → use H4; if both missing → SKIP (note 'no HTF context'). 
- Position: {pos_info}
- 15m snapshot: {snap15}

Policy:
- Focus on **position management, not entry timing.
- Evaluate current P&L (upl), recent volatility (ATR), and market structure (Donchian, RSI, MACD).
- HOLD if the trend remains consistent or the position is within acceptable ATR-based fluctuation.
- CLOSE only if:
  - The unrealized P&L has reached a target profit region (≥ +5%) → take profit;
  - The loss exceeds a tolerable ATR multiple (e.g., >1.5×ATR) → stop loss;
  - Key invalidations occur (EMA flip, MACD divergence, RSI cross 50, Donchian break);
  - Funding premium or liquidity deteriorates significantly (|z|>1.5 adverse or time_to_next_min<120s).
- Favor letting winners run and cutting losers early.
- Be concise and confident; JSON only.
                                                           
Return JSON:
{{
  "action":"HOLD|CLOSE",
  "side":"NONE",
  "confidence":0.0,
  "reasons": [
    "≤3 brief reasons combining microtrend, ATR, and position P&L perspective",
    "include one reason about daily target or ATR-based threshold"],
  "invalidation": ["list invalidation triggers if suggesting CLOSE, e.g., 'hit_profit_target','atr_stop_loss','ema_flip','rsi_cross','funding_gate'"]
}}
""")

_position_timing_chain = (
    _position_timing_prompt.partial(format_instructions=_format_instructions)
    | get_chat_model(task="timing") 
    | _timing_parser
    )

async def run_position_timing_agent(inst, rd: RDState, snap15, snap30=None, positions=None) -> TimingOut:
    raw = await _position_timing_chain.ainvoke({
            "inst": inst,
            "regime": rd.regime, "regime_conf": rd.regime_confidence,
            "direction": rd.direction, "dir_conf": rd.direction_confidence,
            "snap15": snap15.model_dump(),
            "snap30": snap30.model_dump() if snap30 else None,
            "pos_info": format_position_for_prompt(positions, inst),
        }, 
    )
    return raw
