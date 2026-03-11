import asyncio
from typing import Dict, Any, List, Literal, Optional
from pydantic import Field, BaseModel

from trading.models import Position

from agent.chains import create_agent_chain
from agent.schemas import FeatureFrame
from agent.agent_hub import safe_float

from .trend_agent import TrendOutput
from .trigger_agent import build_snapshot_for_trigger

from datetime import datetime
file_name_ts = datetime.now().strftime("%Y%m%d%H%M%S")


class EntryOutput(BaseModel):
    action: Literal[
        "OPEN_LONG", 
        "OPEN_SHORT", 
        "STALK"
    ] = Field(description="The immediate tactical decision.")
    
    trade_thesis: str = Field(
        description="A concise name or summary of the setup (e.g., 'Pre-breakout Coil Accumulation')."
    )
    
    reasoning: str = Field(
        description="Explain exactly why this trade makes sense right now for a 1-7h hold. Synthesize Macro Regime and Micro-Flow."
    )
    
    suggested_leverage: int = Field(
        description="Integer from 1 to 10. Scale leverage based on the quality and conviction of the setup."
    )
    
    risk_invalidation: str = Field(
        description="At what exact structural shift is this thesis proven wrong? (e.g., '15m close below 1H EMA + 0.5% buffer')."
    )


ENTRY_PROMPT = """
Role: Tactical Swing Hunter.
Objective: **MAXIMIZE REALIZED NET PnL WITH POSITIVE EXPECTANCY.** You hunt for asymmetric 1-7 hour moves. Passive observation doesn't pay, but reckless trading destroys statistical edge. Keep it simple: identify the regime, confirm with structure and flow, enter with controlled leverage, and define where you are structurally wrong.

# 1. TRADING HEURISTICS (Mental Models)
*Use these to synthesize the market structure, not to create narratives.*

- **Momentum & Trend:** Only join momentum when higher-timeframe structure and expansion confirm continuation. Avoid late parabolic entries without pullback or structural base.
- **Reversion & Chop:** Fade extremes ONLY at structural range boundaries with confirmation. Do not anticipate reversals in mid-range.
- **Squeeze & Expansion:** Volatility compression alone is not a signal. Enter only AFTER expansion confirms direction.
- **Flow Context:** Order flow must align with structure. Flow alone is not a trade.

# 2. TACTICAL CONSTRAINTS
1. **Cost & Edge Requirement:** We use Market Orders. The expected 1h-7h move must clearly exceed trading costs (fees + slippage) AND provide a minimum Risk:Reward ≥ 1.8 relative to your defined stop.
2. **Leverage Sizing (Capital Preservation First):**
   - *1x - 2x:* Chop, counter-trend, or unclear structure.
   - *3x - 4x:* Trend-aligned continuation with confirmed structure.
   - *5x MAX:* Clean breakout + structural confirmation + macro alignment.
   Never exceed 5x.
3. **Time Horizon:** 1 to 7 Hours. We trade structural swings, not 15-minute noise.

# 3. DECISION PROCESS
**Step 1: Edge Validation**
- Does structure + regime create a clear directional edge for 1-7h?
- Is projected target distance ≥ 1.8 × stop distance?
If no → STALK.

**Step 2: Thesis & Leverage**
- State clearly WHY you are entering now (`trade_thesis`).
- Assign `suggested_leverage` (1-5) based on structural clarity, not emotion.

**Step 3: Risk Definition**
- Define `risk_invalidation` using structural levels with buffer.
- Stop must sit beyond noise, not inside it.

=========================================
# 4. CURRENT DYNAMIC DATA
=========================================

**STRATEGIC CONTEXT:**
- Market Regime: {trend_regime}
- Mandate: "{trend_strategic_mandate}"
- Structural Bias: "{trend_structural_bias}"

**MARKET DATA:**
{trigger_snap}

=========================================
# 5. FINAL REMINDER
=========================================
Act logically. No narrative inflation.
If edge is unclear or R:R is insufficient → STALK.

Produce a JSON strictly matching the schema.
- Action: [OPEN_LONG, OPEN_SHORT, STALK]
- trade_thesis
- reasoning
- suggested_leverage (1-5)
- risk_invalidation

{format_instructions}
"""

def get_entry_agent():
    return create_agent_chain(EntryOutput, ENTRY_PROMPT, task_name="trigger")

async def invoke_entry_agent(
    trend_output: TrendOutput,
    snapshot: FeatureFrame,
    # pos_info: List[Position],
) -> EntryOutput:
    trigger_snap = build_snapshot_for_trigger(snapshot)
    agent_chain = get_entry_agent()
    
    inputs = {
        "trend_regime": trend_output.regime,
        "trend_strategic_mandate": trend_output.strategic_mandate,
        "trend_structural_bias": trend_output.structural_bias,
        "trigger_snap": trigger_snap.model_dump()
    }
    try:
        prompt_val = agent_chain.first.invoke(inputs)
        raw_prompt = prompt_val.to_string()
        
        with open(f"data/live_prompts/debug_entry_prompt_{file_name_ts}.txt", "a", encoding="utf-8") as f:
            f.write(f"=== TIME: {trigger_snap.ts_close} ===\n")
            f.write(raw_prompt)
            f.write("\n================================\n")
    except Exception as e:
        pass
    
    out  = await agent_chain.ainvoke({
        "trend_regime": trend_output.regime,
        "trend_strategic_mandate": trend_output.strategic_mandate,
        "trend_structural_bias": trend_output.structural_bias,
        "trigger_snap": trigger_snap.model_dump()
    })
    return out
