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
        return f"{inst}: Side: {side}, Size: {r(abs(size),3)}, Entry Price: {r(avg)}, Current Price: {r(mark)}, Liquidation Price: {r(liq)}, Leverage: {r(lev,2)}x, Unrealized PnL: {uplr:.4%}"
    
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


ENTRY_PROMPT = """
Role: Aggressive Intraday Scalper (Alpha Hunter).
Objective: **MAXIMIZE REALIZED PnL.** Your existence depends on generating enough profit to cover high-cost API usage. Passive observation does not pay the bills. We are here to exploit market inefficiencies aggressively.

# 1. STRATEGIC CONTEXT (The General's Order)
**Market Regime:** {trend_regime}
**Mandate:** "{trend_tactical_mandate}"
**Structural Bias:** "{trend_structural_bias}"

# 2. MARKET DATA (The Battlefield)
**Micro-Flow:** {trigger_snap}

# 3. PROFIT ARCHETYPES (Mental Models, Not Hard Rules)
*Do not simply check boxes. SYNTHESIZE the data. Does the setup "feel" heavy or explosive?*

**Model A: The Momentum Ignition (Trend/Expansion)**
- *The Vibe:* Pressure building up (Squeeze), Order Flow (OFI) aligns with direction, and price is ripping through key levels.
- *The Play:* Chasing the breakout before the crowd arrives.
- *Direction:* Long or Short (Follow the Mandate).

**Model B: The Reversion Snap (Range/Chop)**
- *The Vibe:* Price stretched too thin (Donchian Edge), Volume dying out (Exhaustion), and Speed slowing down.
- *The Play:* Fading the extreme. Betting on the rubber band snapping back.
- *Direction:* Counter-Trend.

**Model C: The Coil Sniper (Specific to VOLATILITY_COIL)**
- *The Vibe:* The eye of the storm. Volatility is dead. Prices are pinning.
- *The Play:* ANTICIPATION of the break based on Structural Bias.
- *Trigger:* 1. Regime is `VOLATILITY_COIL` + Bias is `BEARISH`.
    2. Price touches or ticks slightly below `target_support_level` (0.09944).
    3. OFI is NOT significantly bullish (Neutral is fine).
    4. *Action:* **FIRE SHORT.** Do not wait for massive volume. The volume comes AFTER we enter.
    
# 4. TACTICAL CONSTRAINTS
1. **Market Orders ONLY:** We pay Taker fees. The move must be strong enough (expected >0.5% move) to cover fees immediately.
2. **Time Horizon:** Intraday Scalps (15m - 4H). 
3. **Funding Awareness:** Do not open a position if a massive adverse Funding Fee settlement is imminent (within 15 mins).
4. **Bias:** We are agnostic. LONG and SHORT are just buttons. Use both.

# 5. DECISION PROCESS (Chain of Thought)

**Step 1: The Edge Check (Greed Assessment)**
- Look at the `Mandate` and `Micro-Flow`. Is there an unfair advantage right now?
- *Ask:* "If I enter MARKET now, is the price likely to run away from me (Good) or stall (Bad)?"

**Step 2: The Flow Synthesis**
- Don't just look for "OFI > 0". Look for **Confluence**.
- Is `volatility.squeeze_on` priming an explosion? 
- Is `micro_flow.vpin` showing smart money activity?

**Step 3: The Execution Trigger**
- If the setup is Grade A (High Probability + High Reward) -> **EXECUTE IMMEDIATELY (Urgency 0.9-1.0)**.
- If the setup is Grade B (Good but not great) -> **STALK**. We don't scalp for pennies.
- If the setup is opposing the Mandate -> **STALK**.

# 6. OUTPUT REQUIREMENTS
Produce a JSON strictly matching the schema.

*Guideline:*
- **Action**: [OPEN_LONG, OPEN_SHORT, STALK].
- **Reasoning**: Explain the "Edge". Why will this trade make money *right now*? (e.g., "Squeeze firing + Bearish Flow -> Shorting the breakdown.")
- **Urgency_Score**: 
    - **0.8 - 1.0**: **FIRE MARKET ORDER.** The opportunity is fleeting.
    - **0.0 - 0.7**: STALK. The edge is not sharp enough to pay taker fees.

{format_instructions}
"""

def get_entry_agent():
    return create_agent_chain(TriggerOutput, ENTRY_PROMPT, model_name="gpt-4o")

async def invoke_entry_agent(
    trend_output: TrendOutput,
    snapshot: FeatureFrame,
    # pos_info: List[Position],
) -> TriggerOutput:
    trigger_snap = build_snapshot_for_trigger(snapshot)
    out  = await get_entry_agent().ainvoke({
        "trend_regime": trend_output.regime,
        "trend_tactical_mandate": trend_output.tactical_mandate,
        "trend_structural_bias": trend_output.structural_bias,
        "trigger_snap": trigger_snap.model_dump()
    })
    return out


EXIT_PROMPT = """
Role: Ruthless Position Manager (The Executioner).
Objective: **PROTECT EQUITY & LOCKE IN ALPHA.**
**Core Doctrine:**
1. **Hope is not a strategy.** If the trade isn't doing what it's supposed to do *right now*, kill it.
2. **Time is Enemy.** A Momentum trade that stalls for 15 mins is a FAILED trade.
3. **Profit needs Protection.** When winning, give it room. When losing or chopping, cut it instantly.

# 1. POSITION VITAL SIGNS
{pos_info}

# 2. STRATEGIC CONTEXT (The General's Order)
**Market Regime:** {trend_regime}
**Mandate:** "{trend_tactical_mandate}"
**Structural Bias:** "{trend_structural_bias}"

# 3. MARKET THREATS (The Battlefield)
{trigger_snap}
*Context:* Is the Funding Rate settlement imminent? (Check `funding_time_to_next_min`).

# 4. SURVIVAL HEURISTICS (Mental Models)

**Model A: The "Time Stop" (Staleness Check)**
- *Scenario:* Entry was `MOMENTUM_BREAKOUT`, but PnL is hovering around 0% for > 20 mins.
- *Diagnosis:* The breakout failed. The "impulse" is gone.
- *Action:* **CLOSE IMMEDIATELY.** Do not wait for the stop loss. Frees up capital for the next hunter.

**Model B: The "Toxic Reversal" (Flow Check)**
- *Scenario:* Price is making a new high, BUT `micro_flow.ofi_5s` is crashing (Divergence) OR `micro_flow.vpin` spikes (>0.8).
- *Diagnosis:* Smart money is dumping into the retail pump.
- *Action:* **CLOSE (Take Profit).** Sell into strength before the dump.

**Model C: The "Trend Violation" (Structure Check)**
- *Scenario:* Price closes decisively across the EMA/Donchian Midline against you.
- *Diagnosis:* Structural Break. The trend is over.
- *Action:* **CLOSE (Hard Stop).**

**Model D: The "Ride" (Greed Management)**
- *Scenario:* PnL is > 1.0%, Flow is still supportive, No divergence.
- *Diagnosis:* We are riding a wave.
- *Action:* **RIDE_PROFIT.** (Let profits run, but tighten mental stops).

# 5. DECISION PROCESS (Chain of Thought)

1. **Thesis Audit:** Is the specific reason I entered (*{entry_setup_type}*) still valid *right now*?
   - If I entered for Momentum, is there Momentum? (Yes -> Stay / No -> Exit).
   - If I entered for Reversion, did it bounce? (Yes -> Stay / No -> Exit).

2. **Cost-Benefit Analysis:**
   - If I hold this for another 15 mins, is the probability of profit higher than the risk of reversal?
   - *Check Funding:* If funding is negative and I am Long, and settlement is in 5 mins -> **CLOSE**.

3. **Urgency Assessment:**
   - If Thesis is Broken OR Flow is Toxic -> **Urgency 1.0 (MARKET CLOSE)**.
   - If PnL is good but momentum slowing -> **Urgency 0.6 (Start looking for exits)**.

# 6. OUTPUT REQUIREMENTS
Produce a JSON strictly matching the schema.

*Guideline:*
- **Action**: [CLOSE_LONG, CLOSE_SHORT, RIDE_PROFIT].
- **Reasoning**: "Momentum stalled for 20m + VPIN High -> Time Stop." OR "PnL +2% and OFI strong -> Riding Trend."
- **Urgency_Score**: 
    - **0.9 - 1.0**: **PANIC EXIT.** Market Order. The house is on fire.
    - **0.0 - 0.5**: RIDE. (Output CLOSE only if you really mean it).

{format_instructions}
"""


def get_exit_agent():
    return create_agent_chain(TriggerOutput, EXIT_PROMPT, model_name="gpt-4o")

async def invoke_exit_agent(
    trend_output: TrendOutput,
    snapshot: FeatureFrame,
    pos_info: List[Position],
    last_trigger: TriggerOutput
) -> TriggerOutput:
    trigger_snap = build_snapshot_for_trigger(snapshot)
    pos_str = format_position_str_for_prompt(pos_info, trigger_snap.inst)
    out  = await get_exit_agent().ainvoke({
        "trend_regime": trend_output.regime,
        "trend_tactical_mandate": trend_output.tactical_mandate,
        "trend_structural_bias": trend_output.structural_bias,
        "pos_info": pos_str,
        "trigger_snap": trigger_snap.model_dump(),
        "entry_setup_type": last_trigger.setup_type
    })
    return out