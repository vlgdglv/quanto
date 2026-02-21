import asyncio
from typing import Dict, Any, List, Literal, Optional
from pydantic import Field, BaseModel

from agent.chains import BaseAgentOutput, create_agent_chain
from agent.schemas import FeatureFrame
from agent.agent_hub import safe_float


# =========================================================
#  1. Trend Snapshot (4H / 1H) - 关注宏观、统计、资金面
# =========================================================
def build_snapshot_for_trend(frame: FeatureFrame) -> FeatureFrame:
    row: Dict[str, Any] = frame.features
    
    # 比如：价格在 Donchian Channel 的什么位置 (0~1)
    donchian_pos = -1
    if row['donchian_upper'] != row['donchian_lower']:
        donchian_pos = (row['c'] - row['donchian_lower']) / (row['donchian_upper'] - row['donchian_lower'])

    features_dict = {
        # --- A. 趋势结构 (Trend Structure) ---
        "macro_trend": {
            "price": row['c'],
            "ema_slow_dist_pct": safe_float((row['c'] - row['ema_slow']) / row['c'] * 100, 2),
            "macd_hist": safe_float(row['macd_hist'], 5),
            
            "mom_slope_3h": safe_float(row['s_mom_slope_H180m']), 
            "mom_slope_7h": safe_float(row['s_mom_slope_H420m']),
        },

        # --- B. 关键点位 (Zones) ---
        "zone_structure": {
            "donchian_width_pct": safe_float(row['donchian_width_norm'] * 100, 2),
            "position_in_channel": safe_float(donchian_pos, 2), # 0.9 以上极其危险，0.1 以下由于支撑
            "dist_to_upper": safe_float(row['s_donchian_dist_upper'], 2),
            "dist_to_lower": safe_float(row['s_donchian_dist_lower'], 2),
        },

        # --- C. 市场情绪与资金 (Sentiment & Funding) ---
        "market_state": {
            "rsi_mean_3h": safe_float(row['s_rsi_mean_H180m'], 1), # 平均 RSI 比瞬时 RSI 更稳
            "funding_rate_annual": safe_float(row['funding_annualized'], 2),
            "funding_premium_z": safe_float(row['funding_premium_z'], 2), # Z-score 很有用
            "oi_change_rate": safe_float(row['d_oi_rate'] * 100, 2), # 持仓变化率
            "sentiment_bias": "Bullish" if row['funding_premium'] > 0 else "Bearish"
        }
    }

    return FeatureFrame(
        inst=frame.inst, tf=frame.tf, ts_close=frame.ts_close,
        features=features_dict, kind="STRATEGY frame"
    )


class TrendOutput(BaseModel):
    regime: Literal[
        "TREND_BULL_IMPULSE",       
        "TREND_BEAR_IMPULSE",       
        "RANGE_ACCUMULATION",       
        "RANGE_DISTRIBUTION",       
        "VOLATILITY_COIL",          
        "CHOP_DEATH_ZONE"           
    ] = Field(description="The precise macro phase of the market cycle based on 1H/4H structure.")
    
    structural_bias: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(
        description="The path of least resistance. Who is in control on the higher timeframe?"
    )
    target_support_level: float = Field(
        description="The critical macro floor. Below this, the bullish thesis is invalidated."
    )
    target_resistance_level: float = Field(
        description="The critical macro ceiling. Above this, the bearish thesis is invalidated."
    )
    key_evidence: List[str] = Field(
        description="A bulleted list of the most compelling macroeconomic and flow evidence (e.g., '4H Donchian contracting + OI climbing suggests massive energy building for a downside break')."
    )
    strategic_mandate: str = Field(
        description="The overarching mission for the intraday Trigger Agents. (e.g., 'Macro bias is heavily bearish. Do not catch falling knives. Hunt for micro-exhaustion rallies into resistance to open shorts.')"
    )
    confidence_score: float = Field(description="0.0-1.0. Reliability of this macro setup. <0.5 means the market is highly unpredictable right now.")


TREND_PROMPT_TEMPLATE = """
Role: Chief Investment Officer (Crypto Macro & Flow Desk).
Objective: **DEFINE THE BATTLEFIELD AND PREDICT THE NEXT EXPANSION.** You operate on High Timeframes (1H, 4H). You do not execute trades. Your job is to hand the intraday Snipers (Trigger Agents) a clear map, a directional bias, and a strategic mandate.

# 0. CONTEXT CHECK
Previous View: {last_context}
*Instruction:* Evaluate the continuity of the market. Has the previous regime been confirmed, rejected, or is it evolving? Do not assume a breakout just because time has passed; let the volatility and flow metrics tell you if energy is building or dissipating.

# 1. MACRO INTELLIGENCE
## STRUCTURAL FLOW (Anchor TF: {anchor_tf})
{anchor_snap}
*Synthesis Questions:*
- What is the broader market structure doing? (Higher highs? Lower lows? Stuck in a box?)
- Is Open Interest (OI) supporting the price action, or diverging?

## MOMENTUM & SENTIMENT (Driver TF: {driver_tf})
{driver_snap}
*Synthesis Questions:*
- How are retail and smart money positioned? 
- Are Funding Rates heavily skewed, creating a potential squeeze scenario?
- Is volatility compressing (Coil) or expanding (Impulse)?

# 2. STRATEGIC DEDUCTION (The "Why" Chain)

**Step A: Diagnose the Macro Regime**
- Look beyond the price. Combine volatility, funding, and structure.
- Is it a true `TREND_IMPULSE`, a messy `CHOP_DEATH_ZONE`, or a ticking time bomb `VOLATILITY_COIL`?
- *Rule of Thumb:* If there is no clear edge, it is CHOP. Do not force a narrative.

**Step B: Define the Map (Support/Resistance)**
- Locate the *heavy* macro levels. These are not 15m wicks; these are 1H/4H structural pivots, major moving averages, or Donchian extremes where the regime would structurally break.

**Step C: Formulate the Strategic Mandate**
- Give the intraday Snipers clear marching orders. Tell them *what* to look for, not *how* to click the button.
- *Good Example:* "Bias is Bearish. The market is heavy. Mandate: Fade any micro-rallies into the 1H EMA. Hunt for breakdown setups."
- *Bad Example:* "Market sell right now." (You are the CIO, leave execution to the Snipers).

# 3. OUTPUT GENERATION
Produce a JSON strictly matching the schema.

*Field Guidelines:*
- `regime`: Select strictly from [TREND_BULL_IMPULSE, TREND_BEAR_IMPULSE, RANGE_ACCUMULATION, RANGE_DISTRIBUTION, VOLATILITY_COIL, CHOP_DEATH_ZONE].
- `structural_bias`: Must be BULLISH, BEARISH, or NEUTRAL.
- `key_evidence`: Extract the 2-3 most critical pieces of data from the snaps that prove your thesis.
- `strategic_mandate`: Give a clear, paragraph-length directive to the Trigger Agent on how they should approach the intraday session.

{format_instructions}
"""


def get_trend_agent():
    return create_agent_chain(TrendOutput, TREND_PROMPT_TEMPLATE, task_name="trend")


async def invoke_trend_agent(
    anchor_frame: FeatureFrame, 
    driver_frame: FeatureFrame,
    last_context: Optional[TrendOutput],
) -> TrendOutput:
    anchor_snap = build_snapshot_for_trend(anchor_frame)
    driver_snap = build_snapshot_for_trend(driver_frame)
    
    if last_context:
        last_context = last_context.model_dump()
    else:
        last_context = "This is the first run, cold starting."
    
    trend_output: TrendOutput = await get_trend_agent().ainvoke({
        "anchor_snap": anchor_snap.model_dump(),
        "anchor_tf": anchor_frame.tf,
        "driver_snap": driver_snap.model_dump(),
        "driver_tf": driver_frame.tf,
        "last_context": last_context
    })
    return trend_output

async def main():
    sample_frame_1H = FeatureFrame(
        inst="DOGE-USDT-SWAP",
        tf="1H",
        ts_close="20260210131500",
        features={'instId': 'DOGE-USDT-SWAP', 'tf': '15m', 'ts': '20260210131500', 'o': 0.0955, 'h': 0.0955, 'l': 0.09499, 'c': 0.095, 'ema_fast': 0.095, 'ema_slow': 0.095, 'macd_dif': 0.0, 'macd_dea': 0.0, 'macd_hist': 0.0, 'rsi': 50.0, 'kdj_k': 33.99, 'kdj_d': 44.66, 'kdj_j': 12.64, 'atr': 0.00051, 'rv_ewma': 0.0, 'squeeze_ratio': None, 'squeeze_on': None, 'donchian_upper': 0.0955, 'donchian_lower': 0.09499, 'donchian_width': 0.00051, 'donchian_width_norm': 1.0, 'er': None, 'spread_bp': 1.05, 'ofi_5s': -338.43, 'qi1': -0.262, 'qi5': -0.276, 'microprice': 0.094954, 'cvd': -3365.34, 'kyle_lambda': 0.0, 'vpin': None, 'funding_rate': 8.8e-05, 'funding_rate_ema': 8.9e-05, 'funding_premium': -0.000526, 'funding_premium_ema': -0.000513, 'funding_premium_z': -0.302, 'funding_annualized': 0.0962, 'funding_time': '20260210160000', 'next_funding_time': '20260211000000', 'funding_time_to_next_min': 645.0, 'oi': 716625.11, 'oiCcy': 716625110.0, 'oiUsd': 68079385.45, 'd_oi': 1.69, 'd_oi_rate': 2e-06, 'oi_ema': 716822.18, 's_mom_slope_H60m': 0.0, 's_rsi_mean_H60m': 50.0, 's_rsi_std_H60m': None, 's_spread_bp_mean_H60m': 1.05, 's_ofi_sum_30m': -338.43, 's_cvd_delta_H60m': 0.0, 's_kyle_ema_H60m': 0.0, 's_oi_rate_H60m': 2e-06, 's_mom_slope_H180m': 0.0, 's_rsi_mean_H180m': 50.0, 's_rsi_std_H180m': None, 's_spread_bp_mean_H180m': 1.05, 's_cvd_delta_H180m': 0.0, 's_kyle_ema_H180m': 0.0, 's_oi_rate_H180m': 2e-06, 's_mom_slope_H420m': 0.0, 's_rsi_mean_H420m': 50.0, 's_rsi_std_H420m': None, 's_spread_bp_mean_H420m': 1.05, 's_cvd_delta_H420m': 0.0, 's_kyle_ema_H420m': 0.0, 's_oi_rate_H420m': 2e-06, 's_macd_pos_streak': 0.0, 's_macd_neg_streak': 0.0, 's_squeeze_on_dur': 0.0, 's_donchian_dist_upper': 0.980392, 's_donchian_dist_lower': 0.019608, 's_donchian_mid_dev': -0.480392},
        kind='FeaturesUpdated'
    )

    sample_frame_4H = FeatureFrame(
        inst="DOGE-USDT-SWAP",
        tf="4H",
        ts_close="20260210130000",
        features={'instId': 'DOGE-USDT-SWAP', 'tf': '30m', 'ts': '20260210130000', 'o': 0.09538, 'h': 0.09551, 'l': 0.09499, 'c': 0.095, 'ema_fast': 0.095, 'ema_slow': 0.095, 'macd_dif': 0.0, 'macd_dea': 0.0, 'macd_hist': 0.0, 'rsi': 50.0, 'kdj_k': 33.97, 'kdj_d': 44.66, 'kdj_j': 12.61, 'atr': 0.00052, 'rv_ewma': 0.0, 'squeeze_ratio': None, 'squeeze_on': None, 'donchian_upper': 0.09551, 'donchian_lower': 0.09499, 'donchian_width': 0.00052, 'donchian_width_norm': 1.0, 'er': None, 'spread_bp': 1.05, 'ofi_5s': -338.43, 'qi1': -0.262, 'qi5': -0.276, 'microprice': 0.094954, 'cvd': -3365.34, 'kyle_lambda': 0.0, 'vpin': None, 'funding_rate': 8.8e-05, 'funding_rate_ema': 8.9e-05, 'funding_premium': -0.000526, 'funding_premium_ema': -0.000513, 'funding_premium_z': -0.302, 'funding_annualized': 0.0962, 'funding_time': '20260210160000', 'next_funding_time': '20260211000000', 'funding_time_to_next_min': 660.0, 'oi': 716625.11, 'oiCcy': 716625110.0, 'oiUsd': 68079385.45, 'd_oi': 1.69, 'd_oi_rate': 2e-06, 'oi_ema': 716822.18, 's_mom_slope_H60m': 0.0, 's_rsi_mean_H60m': 50.0, 's_rsi_std_H60m': None, 's_spread_bp_mean_H60m': 1.05, 's_ofi_sum_30m': -338.43, 's_cvd_delta_H60m': 0.0, 's_kyle_ema_H60m': 0.0, 's_oi_rate_H60m': 2e-06, 's_mom_slope_H180m': 0.0, 's_rsi_mean_H180m': 50.0, 's_rsi_std_H180m': None, 's_spread_bp_mean_H180m': 1.05, 's_cvd_delta_H180m': 0.0, 's_kyle_ema_H180m': 0.0, 's_oi_rate_H180m': 2e-06, 's_mom_slope_H420m': 0.0, 's_rsi_mean_H420m': 50.0, 's_rsi_std_H420m': None, 's_spread_bp_mean_H420m': 1.05, 's_cvd_delta_H420m': 0.0, 's_kyle_ema_H420m': 0.0, 's_oi_rate_H420m': 2e-06, 's_macd_pos_streak': 0.0, 's_macd_neg_streak': 0.0, 's_squeeze_on_dur': 0.0, 's_donchian_dist_upper': 0.980769, 's_donchian_dist_lower': 0.019231, 's_donchian_mid_dev': -0.480769},
        kind='FeaturesUpdated'
    )

    snap_4h = build_snapshot_for_trend(sample_frame_4H)
    snap_1h = build_snapshot_for_trend(sample_frame_1H)
    # print(snap_4h.model_dump())
    # print(snap_1h.model_dump())

    dummy_trend_analysis = TrendOutput(
        regime="VOLATILITY_EXPANSION",
        structural_bias="4H trend is transitioning from sideways to vertical; 1H price action shows a series of higher lows with RSI breaking above 70 without immediate exhaustion.",
        key_evidence=[
            "BBWidth (20, 2) expanded from 0.02 to 0.08 in 4 bars",
            "Volume-to-Average-Volume Ratio > 2.5x",
            "Price > Upper Donchian Channel (20) with strong close"
        ],
        risk_factors=[
            "15m RSI in deep overbought territory (RSI > 85)",
            "Potential liquidity gap below the breakout point at $42,500",
            "Open Interest (OI) rising too rapidly, suggesting speculative FOMO"
        ],
        confidence_score=0.88,
        confidence_reasoning="High confidence due to the confluence of volume expansion and price breaking a multi-week consolidation range with institutional-sized buy walls visible in the order book.",
        tactical_mandate="AGGRESSIVE LONG. Buy 15m pullbacks to the 9-period EMA. Do not wait for deep retracements to the 4H mean as momentum is too high.",
        confirmation_trigger="15m candle closes above previous high with positive Delta in the footprint chart.",
        invalidation_trigger="Price closes back inside the previous 4H range (False Breakout) or 1H candle closes below the 20-period EMA."
    )

    # print(dummy_trend_analysis.model_dump())
    trendor = get_trend_agent()
    # print(trendor)

    out = await trendor.ainvoke({
        "last_context": dummy_trend_analysis.model_dump(),
        "snap4h": snap_4h.model_dump(),
        "snap1h": snap_1h.model_dump(),
    })

    print(out.model_dump())


if __name__ == "__main__":
    asyncio.run(main())