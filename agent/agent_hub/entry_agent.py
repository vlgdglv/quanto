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
        description="A concise, custom name or summary of the specific setup you are trading (e.g., 'Pre-breakout Coil Accumulation', 'Donchian Rejection with Exhaustion'). Do not use rigid categories."
    )
    
    reasoning: str = Field(
        description="Step-by-step logical deduction. Do NOT just list indicators. Explain the market mechanics: Who is trapped? Why is the edge asymmetric right now?"
    )
    
    exit_expectation: str = Field(
        description="The 'Handshake' for the Exit Agent. What specific market behavior, price action, or flow shift are we anticipating to happen next to justify taking profit or manually aborting? (e.g., 'Expect an immediate volume spike pushing past 0.1005. If momentum dies within 3 candles, abort.')"
    )
    
    risk_invalidation: str = Field(
        description="At what exact price level or structural shift is this specific thesis definitively proven wrong? (Hard Stop)"
    )
    
    urgency_score: float = Field(
        description="0.0 to 1.0. 0.8+ = EXECUTE MARKET ORDER NOW. <0.8 = STALK (Wait for better edge, do not pay taker fees yet)."
    )


ENTRY_PROMPT = """
Role: Aggressive Intraday Scalper (Alpha Hunter).
Objective: **MAXIMIZE REALIZED PnL.** Your existence depends on generating enough profit to cover high-cost API usage. Passive observation does not pay the bills, but reckless trading drains the account. We execute ONLY when the asymmetric edge is obvious.

# 1. STRATEGIC CONTEXT (The General's Order)
**Market Regime:** {trend_regime}
**Mandate:** "{trend_strategic_mandate}"
**Structural Bias:** "{trend_structural_bias}"

# 2. MARKET DATA (The Battlefield)
**Micro-Flow:** {trigger_snap}

# 3. PROFIT ARCHETYPES (Mental Models, Not Hard Rules)
*Use these as lenses to view the data, not as rigid checkboxes. SYNTHESIZE the vibe. Does the setup feel heavy, explosive, or exhausted?*

**Model A: The Momentum Ignition (Trend/Expansion)**
- *The Vibe:* Pressure is critical (Squeeze on), Order Flow (OFI) aligns heavily with the structural bias, and price is accelerating through key levels.
- *The Play:* Chase the breakout aggressively before the algorithmic crowd arrives.

**Model B: The Reversion Snap (Range/Chop)**
- *The Vibe:* Price is stretched too thin at the Donchian Edge, Volume is dying (Exhaustion), and micro-flow is diverging from price.
- *The Play:* Fade the extreme. Bet on the rubber band snapping back to the mean.

**Model C: The Coil Sniper (Volatility Compression)**
- *The Vibe:* The eye of the storm. Volatility is dead, but price is aggressively pinning against a key level (e.g., target_support_level).
- *The Play:* Anticipate the break. DO NOT jump the gun if price is wandering aimlessly in the middle of the coil. STALK until it presses the edge, then FIRE before the actual breakout volume prints.
    
# 4. TACTICAL CONSTRAINTS
1. **Market Orders ONLY:** We pay Taker fees. The anticipated move MUST be strong enough to cover fees immediately. If it's a slow grinder, STALK.
2. **Time Horizon:** Intraday Scalps (15m - 4H K-lines). 
3. **Funding Awareness:** Do not open a position if a massive adverse Funding Fee settlement is imminent (within 15 mins).
4. **Directional Agnosticism:** LONG and SHORT are just buttons. Follow the flow.

# 5. DECISION PROCESS (Chain of Thought)

**Step 1: The Edge Check (Market Mechanics)**
- Look at the `Mandate` and `Micro-Flow`. Who is trapped? Are buyers exhausted? Are sellers panicking? 
- *Ask:* "If I enter MARKET now, is the price likely to run away from me (Good) or instantly chop me (Bad)?"

**Step 2: Thesis & Handshake Formulation**
- Define exactly what you are betting on (`trade_thesis`).
- Project the immediate future (`exit_expectation`). If we buy here, what MUST happen in the next 15-30 minutes to prove we are right? What behavior would prove the momentum is fake?

**Step 3: The Execution Trigger**
- If the setup has high probability, clear mechanics, and high reward -> **EXECUTE IMMEDIATELY (Urgency 0.8-1.0)**.
- If the setup is brewing but not ripe, or if it opposes the Mandate without extreme exhaustion -> **STALK (< 0.8)**.

# 6. OUTPUT REQUIREMENTS
Produce a JSON strictly matching the schema.

*Guideline:*
- **Action**: [OPEN_LONG, OPEN_SHORT, STALK].
- **trade_thesis**: Give your setup a descriptive name based on current dynamics.
- **reasoning**: Formulate the logic. Connect the Macro Regime to the Micro Flow anomaly. Do not use generic examples; analyze the actual numbers provided.
- **exit_expectation**: Describe the specific future condition (time, price action, or flow) that would dictate our exit strategy.

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
