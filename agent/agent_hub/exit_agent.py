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
Objective: **PROTECT EQUITY & PRESERVE POSITIVE EXPECTANCY.** Manage risk objectively. No hope. No storytelling.

# 1. CRITICAL RULES
1. Read Net PnL exactly as given.
2. Compare Current Price precisely with `entry_risk_invalidation`.
3. No fabricated logic beyond provided data.

# 2. POSITION MANAGEMENT LOGIC

**Model A: Structural Profit Protection**
- If Net PnL ≥ 2R relative to initial risk and structure weakens significantly → CLOSE.
- If Net PnL ≥ 3R → consider closing to lock structural gain.

**Model B: Hard Stop**
- If `entry_risk_invalidation` is breached → CLOSE IMMEDIATELY.

**Model C: Extreme Flow Breakdown**
- Only exit early if flow is violently hostile AND structure weakens simultaneously.

**Model D: Structural Continuation**
- If structure holds and PnL is within normal swing fluctuation → HOLD.
- Minor 15m pullbacks are not exit signals.

# 3. DECISION PROCESS

Step 1: Check hard stop.
Step 2: Evaluate R-multiple.
Step 3: Assess structural alignment.
If none of the exit conditions are met → HOLD.

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
Let winners reach structural targets.

Produce a JSON strictly matching the schema.
- Action: [CLOSE_LONG, CLOSE_SHORT, HOLD]
- thesis_audit
- reasoning

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
        "entry_risk_invalidation": last_trigger.risk_invalidation
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
        "entry_risk_invalidation": last_trigger.risk_invalidation
    })
    return out