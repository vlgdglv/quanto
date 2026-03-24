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
#  2. Trigger Snapshot (15m) - 关注微观流、裸K形态、瞬时异动
# =========================================================
def build_snapshot_for_trigger(frame: FeatureFrame) -> FeatureFrame:
    row: Dict[str, Any] = frame.features

    features_dict = {
        # --- A. 战术价格行为 (Price Action & Shapes) ---
        "price_action": {
            "price": row['c'],
            "atr_pct": safe_float(row['atr_pct'] * 100, 3), # 决定止损宽度的核心：当前 ATR 占价格百分比
            "body_ratio": safe_float(row['body_ratio'], 3), # 实体比例
            "upper_wick_ratio": safe_float(row['upper_wick_ratio'], 3), # 上影线比例 (找做空衰竭)
            "lower_wick_ratio": safe_float(row['lower_wick_ratio'], 3), # 下影线比例 (找做多接针)
            "zclose_vs_ema_fast": safe_float(row['zclose_vs_ema_fast'], 3), # 战术级偏离度，防止追高
        },

        # --- B. 瞬时振荡器 (Immediate Oscillators) ---
        "oscillators": {
            "rsi": safe_float(row['rsi'], 1),
            "kdj_j": safe_float(row['kdj_j'], 1), # 敏感的极值指标
            "macd_hist": safe_float(row['macd_hist'], 6),
        },

        # --- C. 15m 真实微观战斗局势 (Micro-flow Truth) ---
        "micro_battle": {
            "bar_signed_vol_ratio": safe_float(row['bar_signed_vol_ratio'], 3), # 本根 K 线多空谁赢了 [-1, 1]
            "bar_cvd_delta": safe_float(row['bar_cvd_delta'], 3),               # 绝对净流入体积
            "bar_avg_trade_size": safe_float(row['bar_avg_trade_size'], 3),     # 本根 K 线的平均单笔（找机构大单）
            "bar_vpin_mean": safe_float(row['bar_vpin_mean'], 3),               # 毒性/单向冲击概率
            "bar_kyle_mean": safe_float(row['bar_kyle_mean'], 6),               # 盘口脆弱度
        },
        
        # --- D. 流动性与异动警报 (Liquidity & Anomalies) ---
        "risk_anomalies": {
            "spread_bp_mean": safe_float(row['bar_spread_bp_mean'], 2),
            "spread_bp_max": safe_float(row['bar_spread_bp_max'], 2), # 极其重要：15m 内是否发生过流动性真空/拔网线？
            "oi_surge_now": True if abs(row['d_oi_rate']) > 0.005 else False, # 此刻是否有剧烈增减仓
            "squeeze_on": bool(row['squeeze_on'] == 1), # 结合 1H 的 squeeze_dur 看爆发节点
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
        fee_rate = 0.0005 # 0.07%: 0.02% maker 0.05% taker
        
        size = float(getattr(p, "pos", 0.0))
        if abs(size) < 1e-8:
            return None
        side = "LONG" if size > 0 else "SHORT"
        avg  = float(getattr(p, "avgPx", 0.0) or 0.0)
        mark = float(getattr(p, "markPx", 0.0) or 0.0)
        liq  = float(getattr(p, "liqPx", 0.0) or 0.0)
        lev  = float(getattr(p, "lever", 1.0) or 1.0)
        uplr = float(getattr(p, "uplRatio", 0.0) or 0.0)
        now = datetime.now()
        estimated_fee_pct = fee_rate * lev * 2
        net_uplr_pct = uplr - estimated_fee_pct
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
            f"Estimated Net Unrealized PnL (after fee): {net_uplr_pct:.5%}, "
            f"Hold Time: {hold_time_str}"
        )
    filtered = [p for p in positions if inst is None or getattr(p, "instId", None) == inst]
    if not filtered:
        return "NONE"

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
