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


from typing import Optional
from pydantic import BaseModel, Field
from typing import Literal


class EntryOutput(BaseModel):
    action: Literal[
        "OPEN_LONG", 
        "OPEN_SHORT", 
        "STALK"
    ] = Field(description="The immediate tactical decision.")
    
    trade_thesis: str
    
    reasoning: str
    
    suggested_leverage: Optional[int] = Field(
        default=None,
        description="1-10 leverage"
    )
    
    risk_invalidation: Optional[str] = Field(
        default=None,
        description="Invalidation condition"
    )
    
    target_level: Optional[float] = Field(
        default=None,
        description="Entry level"
    )
    
    time_stop_hours: Optional[int] = Field(
        default=None,
        description="Time stop"
    )
    
    setup_type: Optional[str] = Field(
        default=None,
        description="Setup type"
    )
    
    
ENTRY_PROMPT = """
Role: Tactical Swing Hunter.
Objective: **MAXIMIZE REALIZED NET PnL WITH POSITIVE EXPECTANCY.** You hunt for asymmetric intraday moves, typically 1-7 hours. 
Passive observation does not pay, but reckless trading destroys edge. 
Your advantage is contextual judgment: synthesize regime, structure, flow, volatility, and timing into a real tradeable edge.

# 1. TRADING HEURISTICS (Mental Models)
*Use these to synthesize the market, not to manufacture narratives.*

- **Momentum & Trend:** Join momentum only when higher-timeframe structure and current tactical evidence jointly support continuation. Avoid late parabolic entries without either a pullback, a reclaim, or a fresh confirmation.
- **Reversion & Chop:** Fade extremes only when there is evidence of local exhaustion or rejection near meaningful structure. Do not assume reversals from oscillators alone.
- **Squeeze & Expansion:** Compression is potential, not a trade. Enter only when the market begins to reveal direction through price acceptance, flow, or expansion.
- **Flow Context:** Order flow must add value to the thesis. Flow alone is not a trade.
- **Timing Edge:** A directionally correct idea is not enough. There must be a reason this moment is better than waiting.
- **Tactical Trigger Discipline:** Do not over-weight a single 15m bar’s wick, delta, oscillator extreme, or micro-flow burst. These are timing aids, not standalone reasons to trade. A valid entry requires broader structural support from regime, positioning, and path context.

# 2. TACTICAL CONSTRAINTS
1. **Cost & Edge Requirement:** We use Market Orders. The expected 1h-7h move must clearly exceed trading costs (fees + slippage) and provide at least Risk:Reward ≥ 1.8 relative to your defined stop.
2. **Leverage Sizing (Capital Preservation First):**
   - *1x - 3x:* Chop, counter-trend, or lower-clarity setups.
   - *4x - 6x:* Trend-aligned continuation with strong tactical confirmation.
   - *7x MAX:* Exceptionally clean setup where structure, flow, and path quality are unusually aligned.
   Never exceed 9x.
3. **Time Horizon:** 1 to 7 Hours. We trade structural intraday swings, not random 15-minute noise and not slow multi-session hope trades.

# 3. DECISION PROCESS
**Step 1: Edge Validation**
Ask:
- What is the dominant edge right now?
- Why is entry attractive now rather than 15-30 minutes earlier or later?
- Which 2-4 factors matter most here?
- What is the strongest counterargument?
If you cannot answer these clearly, or if the counterargument is nearly as strong as the thesis → STALK.

**Step 1.5: Bar-Noise Check**
- Ask whether the trade still makes sense if the current 15m bar is treated as noisy or only partially informative.
- If the thesis collapses without this single bar’s signal, it is too fragile → STALK.

**Step 2: Freshness & Non-Repetition**
- If recent execution memory is provided in `trigger_snap`, check whether this is merely a repeat of a recent failed idea.
- If this is similar to a recent failed attempt, explicitly identify what is newly improved now: structure, flow, volatility, acceptance, or timing.
- If nothing is materially new → STALK.

**Step 3: Intraday Plausibility**
- Is this likely to resolve within the same intraday rhythm?
- Does the expected path make sense for an intraday trade, or does it require too much patience?
If it likely needs a slow grind or overnight holding → STALK.

**Step 4: Thesis & Leverage**
- State clearly WHY you are entering now (`trade_thesis`).
- Assign `suggested_leverage` (1-9) based on clarity, factor alignment, and path quality — not emotion.

**Step 5: Risk Definition**
- Define `risk_invalidation` using a structural point or condition with buffer.
- The stop must represent where the thesis is materially wrong, not where you become uncomfortable.
- Your target logic must be structural, even if not explicitly output as a separate field.

=========================================
# 4. CURRENT DYNAMIC DATA
=========================================

**STRATEGIC CONTEXT:**
- Market Regime: {trend_regime}
- Mandate: "{trend_strategic_mandate}"
- Structural Bias: "{trend_structural_bias}"

**MARKET DATA:**
{trigger_snap}

# 5. OUTPUT GENERATION
Produce a JSON strictly matching the schema.

Your output must reflect real judgment, not generic templates.

Requirements:
- If edge is weak, stale, repetitive, or poorly timed → `STALK`
- `trade_thesis` should be concise and specific
- `reasoning` must explicitly cover:
  1. the dominant edge,
  2. why now,
  3. the strongest counterargument,
  4. what is newly favorable if this resembles a recent failed idea,
  5. why this should work on an intraday horizon
- `suggested_leverage` should reflect clarity and path quality
- `risk_invalidation` must define where the thesis is wrong

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
