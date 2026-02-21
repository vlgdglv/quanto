import asyncio
from typing import Dict, Any, List, Literal, Optional
from pydantic import Field, BaseModel

from trading.models import Position

from agent.chains import BaseAgentOutput, create_agent_chain
from agent.schemas import FeatureFrame
from agent.agent_hub import safe_float

from .trend_agent import TrendOutput

from datetime import datetime
file_name_ts = datetime.now().strftime("%Y%m%d%H%M%S")

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
    
    def minutes_since_ctime(ctime_str: str, now: datetime) -> str:
        ctime_ms = int(ctime_str)
        ctime_dt = datetime.fromtimestamp(ctime_ms / 1000)
        minutes = int((now - ctime_dt).total_seconds() // 60)
        if minutes < 0:
            minutes = 0
        return f"{max(0, minutes)} min"

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
        now = datetime.now()
        def r(x, n=6):
            return f"{x:.6f}".rstrip("0").rstrip(".")

        inst = getattr(p, "instId", "?")
        hold_time_str = minutes_since_ctime(getattr(p, "cTime"), now)
        # return f"{inst}: Side: {side}, Size: {r(abs(size),3)}, Entry Price: {r(avg)}, Current Price: {r(mark)}, Liquidation Price: {r(liq)}, Leverage: {r(lev,2)}x, Unrealized PnL: {uplr:.4%}"
        return (
            f"{inst}: "
            f"Side: {side}, "
            f"Size: {r(abs(size), 3)}, "
            f"Entry Price: {r(avg)}, "
            f"Current Price: {r(mark)}, "
            f"Liquidation Price: {r(liq)}, "
            f"Leverage: {r(lev, 2)}x, "
            f"Unrealized PnL: {uplr:.5%}, "
            f"Hold Time: {hold_time_str}"
        )
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
Objective: **CAPTURE VOLATILITY.** Target 10% Daily ROI via aggressive, guaranteed entries.
Core Doctrine:
1. **100% Market Orders:** We DO NOT use Limit orders. We pay Taker fees to guarantee fills. Execution certainty is our priority.
2. **The "Spread" Constraint:** Because we use Market Orders, entering in the *middle* of a range is suicide. You must only fire at the absolute extremes (Support/Resistance) or during explosive momentum.
3. **Time is Alpha:** Once the setup is met, hesitation is a SYSTEM FAILURE.

# 1. STRATEGIC CONTEXT (The General's Order)
**Market Regime:** {trend_regime}
**Mandate:** "{trend_tactical_mandate}"
**Confirm Signal:** "{trend_confirmation_trigger}"

# 2. BATTLEFIELD SNAPSHOT (15m/Tick Data)
**My Position:** {pos_info}
*(Net!=0 -> RIDING. Net=0 -> HUNTING.)*

**Micro-Structure Data:**
{trigger_snap}
*Critical Inputs:*
- `micro_flow.ofi_5s`: Order Flow Imbalance.
- `price.dist_to_donchian_lower`: Proximity to support.
- `volatility.squeeze_on`: Volatility compression status.

# 3. DECISION LOGIC (Chain of Thought)

**Step A: Position State Check**
- **IF RIDING (Has Position):**
    - **Win More:** If Trend is strong, HOLD. We hold for 1h-7h. Don't scalp pennies.
    - **Stop Loss:** If `micro_flow.vpin` spikes against me OR invalidation hit -> CLOSE (Market Order).
    - **Take Profit:** If Regime is Range and Price hits the opposite Band -> CLOSE (Market Order).

- **IF HUNTING (No Position):**
    - **Path A: Momentum Breakout (Trend/Expansion Mode)**
        - Condition: Mandate says TREND/EXPANSION + `squeeze_on` is True.
        - Trigger: Price begins to break EMA + OFI > 0.
        - Action: **OPEN_LONG (MARKET)**. Chase the explosion.
        
    - **Path B: Extreme Mean Reversion (Range Mode)**
        - Condition: Mandate says RANGE.
        - Trigger: Price is **TOUCHING** or extremely close to the Donchian Lower Band.
        - Flow Check: We do not need massive OFI, but we need *Absoprtion* (Volume dropping, selling stalling).
        - Action: **OPEN_LONG (MARKET)**. Buy the fear immediately. Do NOT enter if price is in the middle of the range.

**Step B: The Conviction / Urgency Assessment**
Since we only use Market Orders, 'Urgency' defines our willingness to pay the spread right now.
- **SCREAMING (0.8 - 1.0):** Perfect setup at the extreme edge OR Squeeze firing. -> **FIRE MARKET ORDER NOW.**
- **WAITING (0.0 - 0.7):** Price is wandering in the middle. Setup is sloppy. -> **STALK (Do nothing).**

**Step C: The "Gun to Head" Test**
- If you had to pay a 0.05% taker fee right now, is the expected move large enough (1% - 5%) to justify it? 
- If Yes -> EXECUTE.
- If No -> STALK.

# 4. OUTPUT REQUIREMENTS
Produce a JSON strictly matching the schema.

*Guideline:*
- **Action**: [OPEN_LONG, OPEN_SHORT, CLOSE_LONG, CLOSE_SHORT, RIDE_PROFIT, STALK].
- **Reasoning**: "Price at Range Low + Neg Funding -> Market Buy the Support." OR "Squeeze + OFI -> Momentum Market Long."
- **Urgency_Score**: 
    - **0.8 - 1.0**: EXECUTE IMMEDIATELY.
    - **0.0 - 0.7**: STALK. Wait for better extreme pricing.

{format_instructions}
"""

def get_trigger_agent():
    return create_agent_chain(TriggerOutput, TRIGGER_PROMPT_TEMPLATE, task_name="trigger")

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
