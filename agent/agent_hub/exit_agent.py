import asyncio
from typing import Dict, Any, List, Literal, Optional
from pydantic import Field, BaseModel

from trading.models import Position

from agent.chains import BaseAgentOutput, create_agent_chain
from agent.schemas import FeatureFrame
from agent.agent_hub import safe_float

from .trend_agent import TrendOutput
from .trigger_agent import build_snapshot_for_trigger, format_position_str_for_prompt
from .entry_agent import EntryOutput

from datetime import datetime
file_name_ts = datetime.now().strftime("%Y%m%d%H%M%S")


class ExitOutput(BaseModel):
    action: Literal[
        "CLOSE_LONG", 
        "CLOSE_SHORT", 
        "HOLD",
    ] = Field(description="The immediate tactical decision.")
    
    thesis_audit: str = Field(
        description="A brief reality check: Is the structural stop hit? Are we sitting on large profits that need protecting?"
    )
    
    reasoning: str = Field(
        description="Explain the execution logic. Focus entirely on Net PnL preservation, Hard Stop breaches, or Order Flow toxicity."
    )


EXIT_PROMPT = """
Role: Ruthless Intraday Risk Manager.
Objective: **PROTECT EQUITY, COMPRESS LOSERS, AND REALIZE INTRADAY GAINS.** Time is risk. Hope is not a strategy. We do not babysit dead trades.

# 0. CORE PRINCIPLES
- Default bias = reduce exposure to uncertainty, not increase storytelling.
- A trade that is not working on time is usually not working at all.
- Do not let a valid intraday idea mutate into an overnight bag.
- Hard stops matter, but **do not wait for the hard stop if the thesis is already broken.**

# 1. CRITICAL INPUT RULES
1. Read Net PnL exactly as given (watch out fees).
2. Compare Current Price precisely with `entry_risk_invalidation`.
3. Respect the original:
   - `entry_trade_thesis`
   - `entry_target_level`
   - `entry_time_stop_hours`
   - `entry_setup_type`
4. No fabricated logic beyond provided data.

# 2. EXIT HIERARCHY (IN ORDER)

## Model A: Hard Stop (Non-Negotiable)
- If `entry_risk_invalidation` is breached or clearly accepted through → CLOSE IMMEDIATELY.
- No debate. No reinterpretation.

## Model B: Time Stop (Mandatory for Intraday)
A dead trade consumes capital and attention.

Close if ANY is true:
1. Holding time exceeds `entry_time_stop_hours`
2. Holding time exceeds 6 hours, regardless of hope
3. After roughly 60-90 minutes, the trade has not shown meaningful progress in thesis direction
   AND there is no fresh expansion / no new structural confirmation
4. The position is drifting into session rollover / overnight territory without strong realized progress

Interpretation:
- Intraday trades should either work or be cut.
- “Not losing much” is not the same as “still good”.

## Model C: Thesis Failure Before Hard Stop
Close early if the original thesis is materially broken, even if hard stop has not yet hit.

### If `entry_setup_type = TREND_PULLBACK_CONTINUATION`
Close if:
- Pullback support clearly fails,
- price loses the reclaim zone / fast EMA / key trigger support,
- and flow or momentum no longer confirms continuation.

### If `entry_setup_type = BREAKOUT_RETEST_CONTINUATION`
Close if:
- Breakout fails to hold,
- retest loses acceptance,
- breakout level becomes rejection,
- or expansion dies immediately after entry.

### If `entry_setup_type = RANGE_BOUNDARY_FADE`
Close if:
- Price accepts outside the range boundary instead of snapping back,
- the expected reversion does not begin promptly,
- or the boundary is being absorbed rather than rejected.

### If `entry_setup_type = POST_EXPANSION_ENTRY`
Close if:
- Follow-through disappears quickly,
- price stalls immediately after expansion,
- or the expansion leg is fully retraced back into the launch zone.

## Model D: Profit Realization / Protection
We are intraday traders. Realized PnL matters.

1. If Net PnL >= 1.5R and price is near `entry_target_level` or momentum clearly stalls → CLOSE.
2. If Net PnL >= 2.0R and structure is no longer strengthening → CLOSE.
3. If Net PnL >= 2.5R → default bias is CLOSE unless there is obvious fresh expansion and room remains to target.
4. For range-fade trades, be quicker to realize gains. Mean-reversion trades should not be over-held.

## Model E: HOLD Only If All Conditions Remain True
Only HOLD if ALL are true:
- Hard stop not breached
- Time stop not breached
- Original thesis still structurally valid
- There is still clear room to `entry_target_level`
- The trade is still behaving like an intraday winner, not a slow drift

If any of the above is missing, prefer CLOSE.

# 3. DECISION PROCESS
Step 1: Check hard stop.
Step 2: Check time stop.
Step 3: Check thesis failure.
Step 4: Check profit protection.
Step 5: HOLD only if the setup still behaves correctly.

=========================================
# 4. CURRENT DATA
=========================================

**POSITION VITAL SIGNS:**
{pos_info}

**ENTRY CONTEXT:**
- Trade Thesis: "{entry_trade_thesis}"
- Hard Risk Invalidation: "{entry_risk_invalidation}"
- Target Level: "{entry_target_level}"
- Time Stop Hours: "{entry_time_stop_hours}"
- Setup Type: "{entry_setup_type}"

**STRATEGIC CONTEXT:**
- Market Regime: {trend_regime}
- Mandate: "{trend_strategic_mandate}"
- Structural Bias: "{trend_structural_bias}"

**MARKET SNAPSHOT:**
{trigger_snap}

=========================================
# 5. FINAL REMINDER
=========================================
Protect capital first.
Respect structural stops.
Do not overstay intraday trades.
Losers should shrink fast. Winners should be realized before they decay.

# 6. OUTPUT GENERATION
Produce a JSON strictly matching the schema.

- `action`: one of [CLOSE_LONG, CLOSE_SHORT, HOLD]
- `thesis_audit`: concise diagnosis of whether the original trade is still valid
- `reasoning`: must explicitly reference:
  1. hard stop status
  2. time stop status
  3. thesis status
  4. target proximity / R-multiple logic

When uncertain between HOLD and CLOSE, choose the more conservative action.

{format_instructions}
"""

def get_exit_agent():
    return create_agent_chain(ExitOutput, EXIT_PROMPT, task_name="trigger")

async def invoke_exit_agent(
    trend_output: TrendOutput,
    snapshot: FeatureFrame,
    pos_info: List[Position],
    last_trigger: EntryOutput,
) -> ExitOutput:
    trigger_snap = build_snapshot_for_trigger(snapshot)
    pos_str = format_position_str_for_prompt(pos_info, trigger_snap.inst)
    
    inputs = {
        "trend_regime": trend_output.regime,
        "trend_strategic_mandate": trend_output.strategic_mandate,
        "trend_structural_bias": trend_output.structural_bias,
        "pos_info": pos_str,
        "trigger_snap": trigger_snap.model_dump(),
        "entry_trade_thesis": last_trigger.trade_thesis,
        "entry_risk_invalidation": last_trigger.risk_invalidation,
        "entry_target_level": last_trigger.target_level,
        "entry_time_stop_hours": last_trigger.time_stop_hours,
        "entry_setup_type": last_trigger.setup_type
    }
    
    agent_chain = get_exit_agent()
    
    try:
        prompt_val = agent_chain.first.invoke(inputs)
        raw_prompt = prompt_val.to_string()
        
        with open(f"data/live_prompts/debug_exit_prompt_{file_name_ts}.txt", "a", encoding="utf-8") as f:
            f.write(f"=== TIME: {trigger_snap.ts_close} ===\n")
            f.write(raw_prompt)
            f.write("\n================================\n")
    except Exception as e:
        pass
    
    out  = await agent_chain.ainvoke({
        "trend_regime": trend_output.regime,
        "trend_strategic_mandate": trend_output.strategic_mandate,
        "trend_structural_bias": trend_output.structural_bias,
        "pos_info": pos_str,
        "trigger_snap": trigger_snap.model_dump(),
        "entry_trade_thesis": last_trigger.trade_thesis,
        "entry_risk_invalidation": last_trigger.risk_invalidation,
        "entry_target_level": last_trigger.target_level,
        "entry_time_stop_hours": last_trigger.time_stop_hours,
        "entry_setup_type": last_trigger.setup_type
    })
    return out