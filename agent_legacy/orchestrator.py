from typing import Dict, Any, Optional, Callable, List
from pydantic import BaseModel, Field
from enum import Enum
import json
from agent_legacy.chat_models import ChatModelLike
from agent_legacy.schema import (ActionProposal, Direction, DDSOut, EPMOut, RRFOut, SignalScores)

def parse_json(content: str) -> Dict[str, Any]:
    import json, re
    # 粗暴地截取第一个 {...}
    m = re.search(r"\{.*\}", content, re.S)
    if not m:
        raise ValueError("No JSON object found.")
    return json.loads(m.group(0))

PROMPT_RRF_SYSTEM = """
You are the "Regime & Risk Filter" for crypto perpetuals.
Return ONLY JSON:
{"gate":"OPEN|SOFT|HARD_VETO","vol_bucket":"LOW|MID|HIGH","risk_budget":0-1,"budget_floor":0-1,"leverage_cap":int>=1,"soft_flags":[],"veto_reasons":[],"notes":[]}

Rules:
- HARD_VETO only if (funding_time_to_next_min<=3 && spread_bp>=8 && vol_bucket==HIGH). Else default SOFT. Use OPEN only if (spread_bp<=3 && vol_bucket==LOW).
- vol_bucket via donchian_width_norm: <0.25→LOW, 0.25–0.55→MID, >0.55→HIGH (fallback to atr/price if missing).
- Sizing by bucket (before flags): LOW→risk_budget≈0.7, leverage_cap≈11; MID→0.45, 7; HIGH→0.25, 3.
- Reduce for wide spread (>=6bp) and near funding (<=20m); add soft_flags ["wide_spread","near_funding"] as applicable.
- Enforce: risk_budget∈[0,1]; budget_floor∈[0,risk_budget] (default=min(0.05,risk_budget)); leverage_cap>=1 (default=6).
- If inputs are missing/invalid, still return valid JSON with the defaults.
- No explanations. Output the JSON only.
"""
PROMPT_DDS_SYSTEM = """You are the "Directional Decision Synthesizer".
Only return JSON:
{ "direction": "BUY_LONG|SELL_SHORT|HOLD",
  "base_position": float<=1,
  "signal_scores": { "trend": float[-1,1], "flow": float[-1,1], "composite": float[-1,1] },
  "confidence": float[0,1],
  "rationale": string[] }

Rules:
- trend_momentum → trend score; microstructure → flow score.
- If scores conflict or weak → favor HOLD (base_position<=0.1).
- Use positioning (funding, d_oi_rate, s_oi_rate_*) as tie-break and size modifier.
- Never exceed base_position 0.8; no free text outside JSON.
"""
PROMPT_EPM_SYSTEM = """You are the "Execution & Position Manager" for crypto perpetuals.
Return ONLY JSON:
{"action":"BUY_LONG|SELL_SHORT|HOLD","target_position":0-1,"leverage":int>=1,
 "risk_controls":{"entry":{"mode":"MARKET|LIMIT|TWAP_3","limit_slip_bp":int>=0},
                  "stop":{"type":"ATR_MULT|STRUCT","mult":float>0},
                  "take_profit":{"rr_min":float>0,"trail_atr_mult":float>=0},
                  "auto_reduce":{"conf_drop_pct":0-100,"flow_flip":bool}},
 "notes":[]}

Inputs: RRF (gate OPEN|SOFT|HARD_VETO, risk_budget, budget_floor, leverage_cap, soft_flags), 
DDS (direction, base_position, confidence, scores), and current vol/spread.

Rules:
- If RRF.gate == HARD_VETO → HOLD, size 0, leverage 1.
- Else size = clamp( DDS.base_position * max(DDS.confidence,0), RRF.budget_floor, RRF.risk_budget ).
- If action == HOLD → leverage 1; else leverage ≤ RRF.leverage_cap.
- HIGH vol → wider stop (e.g., mult ≥ 1.4) and lower leverage; wide spread or near funding → prefer LIMIT/TWAP_3, add note.
- Output valid JSON even if inputs are partial; never invent unknown keys; keep text short.
"""

# ====== 三段链（签名固定）======
def build_rrf_chain(llm: ChatModelLike):
    def _call(payload: Dict[str, Any]) -> RRFOut:
        msgs = [{"role":"system","content":PROMPT_RRF_SYSTEM},
                {"role":"user","content":payload["rrf_user"]}]
        resp = llm.invoke(msgs, temperature=0)
        fixed = _repair_rrf_payload(parse_json(resp.get("content","")))          # <<=== 修补
        return RRFOut(**fixed)
    return _call

def build_dds_chain(llm: ChatModelLike):
    def _call(payload: Dict[str, Any], rrf: RRFOut) -> DDSOut:
        msgs = [{"role":"system","content":PROMPT_DDS_SYSTEM},
                {"role":"user","content":payload["dds_user"]}]
        resp = llm.invoke(msgs, temperature=0)
        data = parse_json(resp.get("content",""))
        return DDSOut(**data)
    return _call

def _render_epm_user(snapshot: dict, rrf: RRFOut, dds: DDSOut) -> str:

    vol = {
        "atr": getv(snapshot, "snapshot.atr"),
        "donchian_width_norm": getv(snapshot, "volatility_regime.donchian_width_norm"),
    }
    micro = {
        "spread_bp": getv(snapshot, "snapshot.spread_bp", "microstructure.s_spread_bp_mean_H60m"),
    }
    payload = {
        "instId": snapshot.get("instId"),
        "tf": snapshot.get("tf"),
        "ts": snapshot.get("ts"),
        "rrf": rrf.dict(),
        "dds": dds.dict(),
        "volatility": vol,
        "micro": micro,
        "instruction": "Follow system; output JSON only."
    }
    return json.dumps(payload, ensure_ascii=False)


def _repair_epm_payload(data: dict, rrf: RRFOut) -> dict:
    # action
    act = (data.get("action") or "HOLD").upper()
    if act not in ("BUY_LONG","SELL_SHORT","HOLD"): act = "HOLD"
    data["action"] = act
    # size
    try: sz = float(data.get("target_position", 0.0))
    except: sz = 0.0
    sz = max(0.0, min(1.0, sz))
    # 钳到 budget（HARD_VETO 在上游就会设置为 0）
    floor = float(getattr(rrf, "budget_floor", 0.0) or 0.0)
    sz = min(max(sz, floor if act != "HOLD" else 0.0), rrf.risk_budget)
    data["target_position"] = sz
    # leverage
    try: lev = int(data.get("leverage", 1))
    except: lev = 1
    if act == "HOLD": lev = 1
    lev = max(1, min(lev, int(max(1, rrf.leverage_cap))))
    data["leverage"] = lev
    # risk_controls defaults
    rc = data.get("risk_controls") or {}
    entry = rc.get("entry") or {}
    stop = rc.get("stop") or {}
    tp = rc.get("take_profit") or {}
    ar = rc.get("auto_reduce") or {}
    entry.setdefault("mode","LIMIT")
    entry["limit_slip_bp"] = max(0, int(entry.get("limit_slip_bp", 3)))
    stop.setdefault("type","ATR_MULT")
    stop["mult"] = max(0.5, float(stop.get("mult", 1.2)))
    tp["rr_min"] = max(0.5, float(tp.get("rr_min", 1.6)))
    tp["trail_atr_mult"] = max(0.0, float(tp.get("trail_atr_mult", 0.8)))
    ar["conf_drop_pct"] = max(0, min(100, int(ar.get("conf_drop_pct", 40))))
    ar["flow_flip"] = bool(ar.get("flow_flip", True))
    data["risk_controls"] = {"entry":entry, "stop":stop, "take_profit":tp, "auto_reduce":ar}
    # notes
    if "notes" not in data or not isinstance(data["notes"], list):
        data["notes"] = []
    return data

def build_epm_chain(llm: ChatModelLike):
    def _call(payload: Dict[str, Any], rrf: RRFOut, dds: DDSOut) -> EPMOut:
        # 运行时渲染 user 负载（不要再用 payload["epm_user"]）
        snap = payload.get("_raw_snapshot") or {}
        user = _render_epm_user(snap, rrf, dds)
        msgs = [{"role":"system","content":PROMPT_EPM_SYSTEM},
                {"role":"user","content":user}]
        resp = llm.invoke(msgs, temperature=0)
        raw = parse_json(resp.get("content",""))
        fixed = _repair_epm_payload(raw, rrf)
        return EPMOut(**fixed)
    return _call

# --- in orchestrator.py ---

def _repair_rrf_payload(data: dict) -> dict:
    # 1) vol_bucket 缺省
    vb = (data.get("vol_bucket") or "MID").upper()
    if vb not in ("LOW", "MID", "HIGH"):
        vb = "MID"
    data["vol_bucket"] = vb

    # 2) risk_budget ∈ [0,1]，缺省给 0.3
    try:
        rb = float(data.get("risk_budget", 0.3))
    except Exception:
        rb = 0.3
    data["risk_budget"] = max(0.0, min(1.0, rb))

    # 3) budget_floor ≤ risk_budget（若你已采用“软预算器”版本）
    if "budget_floor" in data:
        try:
            bf = float(data.get("budget_floor") or 0.0)
        except Exception:
            bf = 0.0
        bf = max(0.0, min(data["risk_budget"], bf))
        data["budget_floor"] = bf

    # 4) leverage_cap ≥ 1（关键修补点！）
    try:
        lc = int(data.get("leverage_cap", 5))
    except Exception:
        lc = 5
    if lc < 1:
        lc = 1
    data["leverage_cap"] = lc

    # 5) gate（如果你已经切到 OPEN/SOFT/HARD_VETO 三态）
    if "gate" in data:
        g = str(data.get("gate") or "SOFT").upper()
        if g not in ("OPEN", "SOFT", "HARD_VETO"):
            g = "SOFT"
        data["gate"] = g

    # 6) notes/flags 类型修复
    for k in ("notes", "soft_flags", "veto_reasons"):
        if k in data and not isinstance(data[k], list):
            data[k] = [str(data[k])]

    return data


def enforce_guards(epm: EPMOut, rrf: RRFOut, dds: DDSOut) -> EPMOut:
    # 尺寸：软钳制（带地板）。HOLD 时地板无效。
    if epm.action != Direction.HOLD:
        size = max(rrf.budget_floor, min(epm.target_position, rrf.risk_budget))
    else:
        size = 0.0
    # 杠杆：钳到 cap；HOLD 强制 1x
    lev = 1 if (epm.action == Direction.HOLD) else max(1, min(epm.leverage, rrf.leverage_cap))

    return EPMOut(
        action=epm.action if size > 0 else Direction.HOLD,
        target_position=round(size, 4),
        leverage=(1 if size == 0 else lev),
        risk_controls=epm.risk_controls,
        notes=epm.notes or []
    )


class Orchestrator:
    def __init__(self, llm_rrf: ChatModelLike, llm_dds: ChatModelLike, llm_epm: ChatModelLike,
                 now_ts: Callable[[], int]):
        self.rrf = build_rrf_chain(llm_rrf)
        self.dds = build_dds_chain(llm_dds)
        self.epm = build_epm_chain(llm_epm)
        self.now_ts = now_ts

    def run_snapshot(self, snapshot: Dict[str, Any]) -> ActionProposal:
        payload = build_llm_payload(snapshot)

        # 1) RRF 先跑
        rrf = self.rrf(payload)

        # 2) DDS 永远运行（即便 HARD_VETO）
        dds = self.dds(payload, rrf)

        # 3) EPM：只有 HARD_VETO 才硬性 HOLD
        if rrf.gate == "HARD_VETO":
            epm = EPMOut(action=Direction.HOLD, target_position=0.0, leverage=1,
                        notes=["hard_veto"] + (rrf.veto_reasons or []))
        else:
            epm = self.epm(payload, rrf, dds)

        # 4) 统一“工程钳制”（含预算地板与上限）
        # epm = enforce_guards(epm, rrf, dds)

        return ActionProposal(
            instId=snapshot.get("instId"),
            tf=snapshot.get("tf"),
            ts_decision=self.now_ts(),
            action=epm.action,
            target_position=epm.target_position,
            leverage=epm.leverage,
            confidence=getattr(dds, "confidence", 0.0),
            reasons=(getattr(dds, "rationale", []) or []) + (getattr(epm, "notes", []) or [])
        )


def getv(d: dict, *paths: str, default=None) -> Any:
    """从多个候选路径取第一个有效值。路径形如 'snapshot.atr' / 'volatility_regime.donchian_width_norm'。"""
    for p in paths:
        cur = d
        ok = True
        for seg in p.split("."):
            if not isinstance(cur, dict) or seg not in cur:
                ok = False
                break
            cur = cur[seg]
        if ok and cur is not None:
            return cur
    return default

# ====== 将你的 snapshot(dict) → 各段 user 文本的渲染器（只做“打包”，不变更数值）======
def build_llm_payload(snapshot: dict) -> dict:
    # === RRF 关注 ===
    vol = {
        "atr": getv(snapshot, "snapshot.atr"),
        "donchian_width_norm": getv(snapshot, "volatility_regime.donchian_width_norm"),
        # 你这版提供的是 s_squeeze_on_dur，作为 squeeze_on_dur 的代理输入
        "squeeze_on_dur": getv(snapshot, "volatility_regime.s_squeeze_on_dur"),
    }
    pos = {
        "funding_time_to_next_min": getv(snapshot, "snapshot.funding_time_to_next_min"),
        "funding_rate": getv(snapshot, "snapshot.funding_rate"),
        "funding_premium_z": getv(snapshot, "snapshot.funding_premium_z"),
        "oi": getv(snapshot, "snapshot.oi"),
        "d_oi_rate": getv(snapshot, "snapshot.d_oi_rate"),
    }
    micro_meta = {"spread_bp": getv(snapshot, "snapshot.spread_bp", "microstructure.s_spread_bp_mean_H60m")}

    rrf_user = json.dumps({
        "instId": snapshot.get("instId"),
        "tf": snapshot.get("tf"),
        "ts": snapshot.get("ts"),
        "volatility_regime": vol,
        "positioning": pos,
        "microstructure_meta": micro_meta,
        "instruction": "Follow the system; output JSON only."
    }, ensure_ascii=False)

    # === DDS 关注 ===
    tm = {
        "ema_fast": getv(snapshot, "trend_momentum.ema_fast"),
        "ema_slow": getv(snapshot, "trend_momentum.ema_slow"),
        "macd_dif": getv(snapshot, "trend_momentum.macd_dif"),
        "macd_hist": getv(snapshot, "trend_momentum.macd_hist"),
        "rsi": getv(snapshot, "trend_momentum.rsi"),
        "mom_slope_H60m": getv(snapshot, "trend_momentum.s_mom_slope_H60m"),
        "mom_slope_H180m": getv(snapshot, "trend_momentum.s_mom_slope_H180m"),
        "mom_slope_H420m": getv(snapshot, "trend_momentum.s_mom_slope_H420m"),
    }
    micro = {
        "ofi_5s": getv(snapshot, "microstructure.ofi_5s"),
        "cvd": getv(snapshot, "microstructure.cvd"),
        "s_ofi_sum_30m": getv(snapshot, "microstructure.s_ofi_sum_30m"),
        "s_cvd_delta_H60m": getv(snapshot, "microstructure.s_cvd_delta_H60m"),
        "spread_bp": getv(snapshot, "snapshot.spread_bp", "microstructure.s_spread_bp_mean_H60m"),
        "kyle_lambda": getv(snapshot, "microstructure.extra.kyle_lambda"),
        "vpin": getv(snapshot, "microstructure.extra.vpin"),
        "microprice": getv(snapshot, "microstructure.microprice"),
        "qi1": getv(snapshot, "microstructure.qi1"),
        "qi5": getv(snapshot, "microstructure.qi5"),
    }
    dds_user = json.dumps({
        "instId": snapshot.get("instId"),
        "tf": snapshot.get("tf"),
        "ts": snapshot.get("ts"),
        "trend_momentum": tm,
        "microstructure": micro,
        "positioning_hint": {
            "s_oi_rate_H60m": getv(snapshot, "positioning.s_oi_rate_H60m"),
            "s_oi_rate_H180m": getv(snapshot, "positioning.s_oi_rate_H180m"),
            "s_oi_rate_H420m": getv(snapshot, "positioning.s_oi_rate_H420m"),
        },
        "instruction": "Follow the system; output JSON only."
    }, ensure_ascii=False)

    # 轻量覆盖率自检（避免“全 null”被发送）
    flat_values = list(tm.values()) + list(micro.values()) + list(vol.values()) + list(pos.values())
    coverage = sum(v is not None for v in flat_values)
    return {
        "rrf_user": rrf_user,
        "dds_user": dds_user,
        "_raw_snapshot": snapshot,
        "_coverage": coverage,
        "_total_fields": len(flat_values),
    }