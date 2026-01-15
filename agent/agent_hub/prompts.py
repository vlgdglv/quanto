# agents/prompts.py

from typing import Dict, Any
from agent.schemas import FeatureFrame
import pandas as pd
import math


def safe_float(v, ndigits=4):
   if pd.isna(v): return None
   return round(float(v), ndigits)

# =========================================================
#  1. Strategy Snapshot (4H / 1H) - 关注宏观、统计、资金面
# =========================================================
def build_rd_snapshot(frame: FeatureFrame) -> FeatureFrame:
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

# =========================================================
#  2. Timing Snapshot (15m) - 关注微观流、瞬时动量、异动
# =========================================================
def build_timing_snapshot(frame: FeatureFrame) -> FeatureFrame:
    row = pd.Series(frame.features)

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

STRATEGY_PROMPT_TEMPLATE = """
Role: Senior Crypto Market Strategist (Macro & Trend Analysis).

# OBJECTIVE
Analyze the market structure of {inst} using 4H (Macro) and 1H (Trend) smoothed statistical data.
Determine the dominant REGIME to guide the tactical execution bot.

# DATA INPUTS

## 1. 4H MACRO FRAME (The Big Picture)
{snap4h}
*Focus on:* `macro_trend.mom_slope_7h`, `zone_structure.position_in_channel`, `market_state.funding_premium_z`

## 2. 1H TREND FRAME (The Immediate Trend)
{snap1h}
*Focus on:* `macro_trend.mom_slope_3h`, `market_state.rsi_mean_3h`

# ANALYSIS PROTOCOL (Step-by-Step)

1. **Trend Alignment Check**:
   - Compare 4H `mom_slope_7h` and 1H `mom_slope_3h`. Are they aligned?
   - If both Positive -> Potential BULLISH.
   - If both Negative -> Potential BEARISH.
   - If Divergent -> NEUTRAL/CHOP.

2. **Overextension & Crowding Check (The Filter)**:
   - Check `zone_structure.position_in_channel`: Is price > 0.9 (Overbought) or < 0.1 (Oversold)?
   - Check `market_state.funding_premium_z`: Is Z-score > 2.0 (Long Crowded)? 
   - *Crucial*: If Trend is BULLISH but Funding is Crowded/Overbought -> Downgrade to NEUTRAL (Risk of squeeze).

3. **Regime Definition**:
   - **LONG**: Strong Upside Momentum + No Overcrowding.
   - **SHORT**: Strong Downside Momentum + Not Oversold.
   - **NEUTRAL**: Conflicting signals, low momentum, or extreme crowding.

# OUTPUT FORMAT
Return the analysis in JSON.
{format_instructions}
"""


TACTICAL_PROMPT_TEMPLATE = """
Role: High-Frequency Crypto Execution Algo (15m Timeframe).

# 1. CURRENT STATUS
**Position:** "{pos_info}" 
*(Format: net=LONG/SHORT/None size=...)*
**Strategy Context:** {strategy.regime} (Conf: {strategy.confidence})
**Permissions (ROE):**
{strategy_permissions_text}

# 2. TACTICAL DATA (15m Snapshot)
{snapshot}
*Focus Keys:* - `micro_flow` (CVD, VPIN, OFI -> The Truth of Liquidity)
- `small_trend` (Momentum)
- `volatility` (Squeeze status)

# 3. EXECUTION LOGIC (Chain of Thought)

**Step A: Assess Position & Intent**
- If Position is **LONG**:
  - Are Micro-flows (CVD/OFI) turning negative? -> Signal **SELL** (Close).
  - Is Trend still strong? -> Signal **HOLD**.
- If Position is **SHORT**:
  - Are Micro-flows turning positive? -> Signal **BUY** (Close).
  - Is Down-trend persisting? -> Signal **HOLD**.
- If Position is **NONE (Flat)**:
  - Check ROE: Are you allowed to enter?
  - If ROE says LONG_ALLOWED: Look for positive OFI + Price Support -> Signal **BUY**.
  - If ROE says SHORT_ALLOWED: Look for negative OFI + Price Resistance -> Signal **SELL**.

**Step B: Micro-Structure Verification (The Alpha)**
- **VPIN**: High VPIN (> avg) means informed trading. If Price moves against you with High VPIN, EXIT IMMEDIATELY.
- **CVD Divergence**: Price making Highs but CVD making Lows? -> Trap -> Do NOT Enter / Close Position.
- **Squeeze**: If `volatility.squeeze_on` is True, expect explosive move.

**Step C: Final Conflict Check**
- YOU CANNOT Open a NEW position against the ROE (Strategy).
- YOU CAN Close an existing position against the ROE (Risk Management).

# OUTPUT DECISION
Select ONE Action:
- **BUY**: Open Long OR Close Short.
- **SELL**: Open Short OR Close Long.
- **HOLD**: Do nothing / Stay in trade.

Return JSON:
{format_instructions}
"""