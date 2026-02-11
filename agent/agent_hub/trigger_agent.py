import asyncio
from typing import Dict, Any, List, Literal, Optional
from pydantic import Field, BaseModel

from trading.models import Position

from agent.chains import BaseAgentOutput, create_agent_chain
from agent.schemas import FeatureFrame
from agent.agent_hub import safe_float

from .trend_agent import TrendOutput


# =========================================================
#  2. Trigger Snapshot (15m) - 关注微观流、瞬时动量、异动
# =========================================================
def build_snapshot_for_trigger(frame: FeatureFrame) -> FeatureFrame:
    row: Dict[str, Any] = frame.features

    features_dict = {
        # --- A. 瞬时动量 (Immediate Momentum) ---
        "small_trend": {
            "price": row['c'],
            "ema_fast_dist": safe_float((row['c'] - row['ema_fast']), 5),
            "macd_hist": safe_float(row['macd_hist'], 6), # 15m 级别要看精度高一点
            "rsi": safe_float(row['rsi'], 1),
            "kdj_j": safe_float(row['kdj_j'], 1), # KDJ J值反应极快
        },

        # --- B. 微观结构 (Microstructure - YOUR ALPHA) ---
        "micro_flow": {
            "cvd_cumulative": safe_float(row['cvd'], 1),   # 累计成交量差
            "ofi_5s": safe_float(row['ofi_5s'], 2),        # 订单流不平衡 (Order Flow Imbalance)
            "vpin": safe_float(row['vpin'], 2),            # 毒性流 (Informed Trading)
            "kyle_lambda": safe_float(row['kyle_lambda'], 2), # 市场深度/价格冲击成本
            "microprice_delta": safe_float(row['microprice'] - row['c'], 5), # 微观价格偏离
            "spread_bp": safe_float(row['spread_bp'], 2)
        },

        # --- C. 爆发力 (Volatility & Squeeze) ---
        "volatility": {
            "squeeze_on": bool(row['squeeze_on'] == 0), # TTM Squeeze 信号
            "squeeze_ratio": safe_float(row['squeeze_ratio'], 2),
            "atr": safe_float(row['atr'], 4),
            "er": safe_float(row['er'], 2) # Efficiency Ratio (Kama Efficiency)
        },
        
        # --- D. 资金异动 (Flow Anomalies) ---
        # 15m 级别只看剧烈的资金异动
        "anomalies": {
            "oi_surge": True if abs(row['d_oi_rate']) > 0.01 else False, # 突发持仓变化
            "funding_pressure": safe_float(row['funding_rate'], 6)
        }
    }

    return FeatureFrame(
        inst=frame.inst, tf=frame.tf, ts_close=frame.ts_close,
        features=features_dict, kind="TIMING frame"
    )

def format_position_str_for_prompt(positions: List[Position], inst: str) -> str:
    """
    positions: List[Position]
    return: 'NONE' or 'net=LONG/SHORT size avg mark liq lev uplr'
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


class TriggerOutput(BaseModel):
    action: Literal[
        "OPEN_LONG", 
        "OPEN_SHORT", 
        "CLOSE_LONG", 
        "CLOSE_SHORT", 
        "RIDE_PROFIT",
        "STALK"
    ] = Field(description="The immediate tactical decision.")

    setup_type: Literal["MOMENTUM_BREAKOUT", "MEAN_REVERSION_PULLBACK", "SQUEEZE_PANIC", "NONE"] = Field(
        description="The type of opportunity identified."
    )
    
    urgency_score: float = Field(
        description="0.0 to 1.0. How fast is the price moving away? 0.9+ = MUST EXECUTE NOW (Market Order). <0.5 = Patient (Limit Order)."
    )

    reasoning: str = Field(description="Why this action? Combine Macro Regime + Micro Flow.")
    risk_invalidation: str = Field(description="At what price/condition do we admit we are wrong?")


TRIGGER_PROMPT_TEMPLATE = """
Role: High-Frequency Execution Algo (Alpha Predator Mode).
Objective: **CAPTURE VOLATILITY.**
Core Doctrine:
1. **Time is Alpha.** Hesitation = Slippage.
2. **Whipsaws are cost of business.** Missing a 5% pump is a SYSTEM FAILURE.
3. **Prefer tight stops over missed entries.** If the Trend is confirmed, get in.

# 1. STRATEGIC CONTEXT (The General's Order)
**Market Regime:** {trend_regime}
**Mandate:** "{trend_tactical_mandate}" (e.g., "Aggressive Long", "Fade Highs")
**Confirm Signal:** "{trend_confirmation_trigger}"

# 2. BATTLEFIELD SNAPSHOT (15m/Tick Data)
**My Position (pos_info):** {pos_info}
*(Net!=0 -> I am RIDING. Net=0 -> I am HUNTING.)*

**Micro-Structure Data:**
{trigger_snap}
*Critical Inputs:*
- `micro_flow.ofi_5s`: Order Flow Imbalance. Positive = Buying Pressure.
- `volatility.squeeze_on`: If True -> Volatility is compressing, ready to explode.
- `price.dist_to_ema`: Is price overextended or at value?

# 3. DECISION LOGIC (Chain of Thought)

**Step A: Position State Check**
- **IF I HAVE POSITION ({pos_info} != None):**
    - Am I winning? -> **RIDE_PROFIT**. Let profits run until OFI flips or Resistance hit.
    - Is the trade invalid? -> **CLOSE**. (Don't pray, cut toxic flow immediately).
    - *Constraint:* Do not close just because "price stalled". Close only on **Reversal Signals**.

- **IF I HAVE NO POSITION:**
    - I must find an entry **NOW** unless the market is dead.
    - Check Mandate: Does Trend say LONG?
    - Check Micro: Is OFI > 0? Is Price > VWAP?
    - **Conclusion:** If Trend + Micro align -> **OPEN_LONG**. Do not wait for "perfect" pattern. **70% confidence is a GO.**

**Step B: Urgency Assessment (Market vs Limit)**
- **SCREAMING:** Squeeze is ON + High Volume Spike -> **URGENCY 1.0 (Market Order)**. Don't risk missing fill.
- **BUILDING:** Consistent OFI + Grinding Price -> **URGENCY 0.7 (Aggressive Limit)**. Bid at Ask.
- **FADING:** Price pulling back to EMA -> **URGENCY 0.4 (Passive Limit)**. Bid at Support.

**Step C: The "Gun to Head" Test**
- If you HAD to trade right now, is the R:R (Risk/Reward) favorable?
- If Yes -> **EXECUTE**.
- Only Output "STALK" if:
    1. Trend and Micro flow actively CONTRADICT each other.
    2. Volatility is zero (Dead Market).

# 4. OUTPUT REQUIREMENTS
Produce a JSON strictly matching the schema.

*Guideline:*
- **Action**: [OPEN_LONG, OPEN_SHORT, CLOSE_LONG, CLOSE_SHORT, RIDE_PROFIT, STALK].
- **Reasoning**: Be brief. "Mandate Long + OFI Positive -> Entering."
- **Urgency_Score**: 
    - **0.8 - 1.0**: IMMINENT BREAKOUT. Pay Taker fees.
    - **0.5 - 0.7**: STRONG FLOW. Join the bid.
    - **0.0 - 0.4**: PATIENT ENTRY.

{format_instructions}
"""

# TRIGGER_PROMPT_TEMPLATE = """
# Role: High-Frequency Alpha Predator (Crypto Perpetual Futures).
# Objective: **Aggressive Capital Deployment.** Constraint: Missing a high-volatility move is considered a SYSTEM FAILURE. We prefer small stop-losses over missed profits.

# # 1. STRATEGIC COMMAND (The General's Order)
# **Regime:** {trend_regime}
# **Tactical Mandate:** "{trend_tactical_mandate}"
# **Trigger Condition:** "{trend_confirmation_trigger}"

# # 2. BATTLEFIELD INTEL (15m Snapshot)
# **Current Status:** {pos_info}
# *(Net=NONE -> You are STALKING. Net!=NONE -> You are RIDING.)*

# **Micro-Flow Data (The Truth):**
# {trigger_snap}
# *Critical Alpha Signals:*
# - `micro_flow.ofi_5s`: If > 0 and Price rising -> **URGENT BUY**.
# - `volatility.squeeze_on`: If True -> **EXPLOSION IMMINENT**. Do not use Limit orders.
# - `micro_flow.vpin`: If High -> **TOXIC FLOW**. Smart money is exiting/reversing.

# # 3. DECISION LOGIC (Chain of Thought)

# **Phase A: The "Urgency" Assessment (Market vs Limit)**
# - Is `volatility.squeeze_on` TRUE? OR Is `micro_flow.ofi_5s` spiking massively?
#   -> **URGENCY = HIGH (0.9-1.0)**. Use MARKET orders. Price will run away.
# - Is price drifting to EMA/Donchian Support?
#   -> **URGENCY = LOW (0.3-0.5)**. Use LIMIT orders. Patiently wait for fill.

# **Phase B: Action Selection (Eliminate Hesitation)**

# IF STALKING (No Position):
# 1.  **MOMENTUM_BREAKOUT:** Strategy says BULL/EXPANSION + Price breaks resistance + High Urgency. -> **OPEN_LONG**.
# 2.  **MEAN_REVERSION_PULLBACK:** Strategy says BULL + Price drops to Support + Low Urgency. -> **OPEN_LONG**.
# 3.  **SQUEEZE_PANIC:** Volatility Squeeze firing + Flow aligns. -> **OPEN_LONG/SHORT IMMEDIATELY**.
# 4.  *Note:* Only choose "STALK" if the setup is barely forming. **Do not waiting for "perfect" certainty. 70% certainty is enough.**

# IF RIDING (Has Position):
# 1.  **RIDE_PROFIT:** Trend is strong. Flow confirms direction. -> **RIDE_PROFIT** (Do not sell early!).
# 2.  **CLOSE (Stop/Take Profit):** - Flow Reversal (OFI flips against you).
#     - VPIN Spike (Toxic exit).
#     - Strategic Invalidation reached.

# # 4. OUTPUT REQUIREMENTS
# Produce a JSON decision.
# - **Action**: Select strictly from [OPEN_LONG, OPEN_SHORT, CLOSE_LONG, CLOSE_SHORT, RIDE_PROFIT, STALK].
# - **Urgency_Score**: 
#     - **0.9 - 1.0**: "I need to get in/out THIS SECOND." (Aggressive Taker)
#     - **0.6 - 0.8**: "Get in soon." (Aggressive Maker / Marketable Limit)
#     - **0.0 - 0.5**: "Waiting for price to come to me." (Passive Maker)

# {format_instructions}
# """

def get_trigger_agent():
    return create_agent_chain(TriggerOutput, TRIGGER_PROMPT_TEMPLATE, model_name="gpt-4o")

async def invoke_trigger_agent(
    trend_output: TrendOutput,
    snapshot: FeatureFrame,
    pos_info: List[Position],
) -> TriggerOutput:
    trigger_snap = build_snapshot_for_trigger(snapshot)
    pos_str = format_position_str_for_prompt(pos_info, trigger_snap.inst)
    out  = await get_trigger_agent().ainvoke({
        "trend_regime": trend_output.regime,
        "trend_tactical_mandate": trend_output.tactical_mandate,
        "trend_confirmation_trigger": trend_output.confirmation_trigger,
        "pos_info": pos_str,
        "trigger_snap": trigger_snap.model_dump()
    })
    return out