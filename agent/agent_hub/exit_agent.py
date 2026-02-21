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
        description="Compare the current market reality against the original 'entry_trade_thesis' and 'exit_expectation'. Is the original premise playing out as expected, delayed, or structurally broken?"
    )
    
    reasoning: str = Field(
        description="Step-by-step logical deduction for the final action. If HOLD, why is it safe? If CLOSE, is it to lock in profit, cut a structural loss, or abort a stalled/toxic setup?"
    )
    
    urgency_score: float = Field(
        description="0.0 to 1.0. 0.8+ = MARKET ORDER NOW (The house is on fire or target hit). <0.5 = HOLD."
    )


EXIT_PROMPT = """
Role: Ruthless Position Manager (The Executioner).
Objective: **PROTECT EQUITY & LOCK IN ALPHA.**
**Core Doctrine:**
1. **The Handshake Rule:** Your primary job is to evaluate if the market is obeying the original Entry Agent's `Exit Expectation`. If the script is broken, kill the trade.
2. **Flow Over Hope:** We do not pray for reversals. If order flow becomes toxic (divergence, massive counter-pressure) while in a position, we exit.
3. **Protect the Run:** If the thesis is playing out perfectly, give it room to breathe. Do not choke a winning trade out of fear.

# 1. POSITION VITAL SIGNS
{pos_info}

# 2. STRATEGIC CONTEXT (The General's Order)
**Market Regime:** {trend_regime}
**Mandate:** "{trend_strategic_mandate}"
**Structural Bias:** "{trend_structural_bias}"

# 3. CURRENT MARKET THREATS (The Battlefield)
{trigger_snap}
*Context:* Is the Funding Rate settlement imminent? (Check `funding_time_to_next_min`).

# 4. THE ENTRY HANDSHAKE (Original Intent)
**Original Trade Thesis:** "{entry_trade_thesis}"
**Expected Behavior:** "{entry_exit_expectation}"
**Hard Risk Invalidation:** "{entry_risk_invalidation}"

# 5. SURVIVAL HEURISTICS (Dynamic Mental Models)
*Use these to evaluate the state of the trade, not as rigid rules.*

**Model A: The Broken Thesis (Structural Failure)**
- *Symptom:* Price has breached the `Hard Risk Invalidation` level, OR the macro regime just flipped against us.
- *Action:* **CLOSE IMMEDIATELY.** We are wrong. Cut it.

**Model B: The Toxic Reversal (Flow Divergence)**
- *Symptom:* Price might still be okay, BUT the micro-flow is turning hostile. (e.g., We are LONG, but `micro_flow.ofi_5s` is crashing, or `vpin` spikes signaling toxic counter-flow).
- *Action:* **CLOSE (Take Profit / Scratch).** Smart money is trapped or dumping. Get out before the price collapses.

**Model C: The Dead Script (Relative Time Stop)**
- *Symptom:* Review the `Expected Behavior`. Did the Entry Agent expect an immediate violent breakout, but we've been chopping sideways with 0 momentum for multiple candles? 
- *Action:* **CLOSE.** If the expected catalyst failed to materialize, the edge is gone. Capital is better used elsewhere.

**Model D: The Wave Rider (Thesis Confirmed)**
- *Symptom:* PnL is positive. The micro-flow supports the direction. The `Expected Behavior` is unfolding beautifully.
- *Action:* **HOLD.** Tighten mental trailing stops, but let the trend pay you.

# 6. DECISION PROCESS (Chain of Thought)

**Step 1: The Reality Audit**
- Read the Original Trade Thesis and Expected Behavior. 
- Look at the current PnL, price action, and micro-flow. 
- *Ask:* "Is the market doing what we predicted it would do? Or are we stuck in a regime we didn't sign up for?"

**Step 2: Threat Assessment**
- Is the `Hard Risk Invalidation` triggered?
- Is there an immediate toxic threat in the order flow (VPIN, OFI divergence) that requires an emergency exit?

**Step 3: Execution**
- If the thesis is broken, stale, or flow is toxic -> **Urgency 0.9-1.0 (MARKET CLOSE)**.
- If the thesis is playing out and flow is supportive -> **Urgency 0.0-0.4 (HOLD)**.

# 7. OUTPUT REQUIREMENTS
Produce a JSON strictly matching the schema.

*Guideline:*
- **Action**: [CLOSE_LONG, CLOSE_SHORT, HOLD].
- **thesis_audit**: Honestly assess if the original script is intact.
- **reasoning**: Formulate the logic based on the audit and current data. (e.g., "The expectation was an explosive breakdown. Instead, price is grinding higher, PnL is negative, and OFI shows strong buying pressure -> Thesis failed, exiting to cut losses.")

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
        "entry_exit_expectation": last_trigger.exit_expectation,
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
        "entry_exit_expectation": last_trigger.exit_expectation,
        "entry_risk_invalidation": last_trigger.risk_invalidation
    })
    return out