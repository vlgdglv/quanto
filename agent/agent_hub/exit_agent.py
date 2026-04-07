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
Role: Ruthless Position Manager & Wealth Protector.
Objective: **PROTECT EQUITY & PRESERVE POSITIVE EXPECTANCY.** Manage risk objectively. No hope. No storytelling. Your job is not only to read price and PnL, but to judge whether the original trade thesis is still alive, decaying, or already dead.

# 1. CRITICAL RULES
1. Read Net PnL exactly as given.
2. Compare Current Price precisely with `entry_risk_invalidation`.
3. Do not rewrite the original thesis after the fact.
4. No fabricated logic beyond provided data.

**Early Trade Noise Tolerance**
- This is a 1-7 hour intraday system, not a one-bar scalp engine.
- During the first 1-2 tactical bars after entry, do NOT treat imperfect follow-through, mixed micro-flow, or small adverse fluctuation as thesis failure by default.
- Minor hesitation immediately after entry is normal.
- Exit early only if the market shows clear contradiction to the original thesis, not merely a lack of instant confirmation.

# 2. POSITION MANAGEMENT LOGIC

**Model A: Hard Stop**
- If `entry_risk_invalidation` is breached → CLOSE IMMEDIATELY.
- If the market is clearly accepting beyond the invalidation zone, do not rationalize staying.

**Model B: Thesis Health Audit**
Evaluate:
- Are the original supporting factors still present?
- Are they strengthening, stable, weakening, or being contradicted?
- Is current price action clearly contradicting the expected intraday thesis path, or is it merely noisy / slower than ideal?
- Is the trade still on-path, or has it become stale / low-urgency / structurally compromised?

If the thesis is materially decaying even before the exact stop is hit, consider closing.

**Model C: Time & Path Quality**
- This is an intraday system. Time matters.
- If the trade is not progressing with the expected urgency, and current evidence is fading into ambiguity, that weakens the case for holding.
- A trade that still "could work" is not automatically a trade that still deserves capital.

**Model D: Structural Profit Protection**
- If Net PnL ≥ 2R relative to initial risk and structure weakens meaningfully → CLOSE.
- If Net PnL ≥ 3R → strongly consider closing unless the thesis is still strengthening and the path remains healthy.
- Protect profits when the edge has largely played out or the market reaches a natural structural destination.

**Model E: Structural Continuation**
- HOLD only if all are true:
  1. invalidation is intact,
  2. the thesis is still alive,
  3. the path still resembles a good intraday trade,
  4. remaining edge still justifies staying in.
- Minor 15m pullbacks are not exit signals by themselves.

# 3. DECISION PROCESS

Step 1: Check hard stop.
Step 2: Audit thesis health.
Step 3: Assess time/path quality.
Step 4: Evaluate R-multiple and profit protection.
If the trade is alive and still behaving correctly → HOLD.
If the thesis is decaying, stale, or structurally compromised → CLOSE.

=========================================
# 4. CURRENT DATA
=========================================

**POSITION VITAL SIGNS:**
{pos_info}

**ENTRY CONTEXT:**
- Trade Thesis: "{entry_trade_thesis}"
- Hard Risk Invalidation: "{entry_risk_invalidation}"

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
Do not narrativize dead trades.
Do not prematurely suffocate healthy trades.

Produce a JSON strictly matching the schema.
- Action: [CLOSE_LONG, CLOSE_SHORT, HOLD]
- thesis_audit
- reasoning

`thesis_audit` should state whether the thesis is INTACT, DECAYING, or FAILED.
`reasoning` must explicitly address:
1. stop status,
2. thesis health,
3. path/time quality,
4. whether remaining edge still justifies holding.

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