# agent/bb_agents.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import json, re, math
from dataclasses import dataclass

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

# ========== 规范化与特征打包（省 token） ==========
DEFAULT_SCALE = {
    "rsi": (50,12), "macd_dif": (0,0.6), "macd_hist": (0,0.6),
    "s_mom_slope_H60m": (0,1.0), "s_mom_slope_H180m": (0,1.0), "s_mom_slope_H420m": (0,1.0),
    "ofi_5s": (0,1.0), "s_ofi_sum_30m": (0,1.0), "s_cvd_delta_H60m": (0,1.0),
    "s_spread_bp_mean_H60m": (0,1.0), "microprice_dev": (0,1.0),
    "s_kyle_ema_H60m": (0,1.0), "vpin": (0,1.0),
    "atr_over_price": (0.005,0.003), "donchian_width_norm": (0.3,0.15),
    "squeeze_on": (0,1.0), "s_squeeze_on_dur": (0,1.0),
    "s_donchian_dist_upper": (0,1.0), "s_donchian_dist_lower": (0,1.0), "s_donchian_mid_dev": (0,1.0),
    "funding_premium_z": (0,1.0), "d_oi_rate": (0,0.02), "s_oi_rate_H60m": (0,1.0), "s_oi_rate_H180m": (0,1.0),
}

CANDIDATES = [
    "rsi","macd_dif","macd_hist","s_mom_slope_H60m","s_mom_slope_H180m","s_mom_slope_H420m",
    "ofi_5s","s_ofi_sum_30m","s_cvd_delta_H60m","s_spread_bp_mean_H60m","microprice_dev",
    "s_kyle_ema_H60m","vpin",
    "atr_over_price","donchian_width_norm","squeeze_on","s_squeeze_on_dur",
    "s_donchian_dist_upper","s_donchian_dist_lower","s_donchian_mid_dev",
    "funding_premium_z","d_oi_rate","s_oi_rate_H60m","s_oi_rate_H180m"
]

def z(v, mean, std, clip=3.5):
    if v is None or std is None or std==0: return 0.0
    z = (v-mean)/std
    if clip: z = max(-clip, min(clip, z))
    return round(z,3)

def build_feature_pack(snap: Dict[str,Any], topk:int=16) -> Dict[str,Any]:
    price = snap.get("c") or snap.get("last_price") or 0.0
    atr = snap.get("atr") or 0.0
    raw = dict(
        rsi=snap.get("rsi"),
        macd_dif=snap.get("macd_dif"),
        macd_hist=snap.get("macd_hist"),
        s_mom_slope_H60m=snap.get("s_mom_slope_H60m"),
        s_mom_slope_H180m=snap.get("s_mom_slope_H180m"),
        s_mom_slope_H420m=snap.get("s_mom_slope_H420m"),
        ofi_5s=snap.get("ofi_5s"),
        s_ofi_sum_30m=snap.get("s_ofi_sum_30m"),
        s_cvd_delta_H60m=snap.get("s_cvd_delta_H60m"),
        s_spread_bp_mean_H60m=snap.get("s_spread_bp_mean_H60m"),
        microprice_dev= snap.get("microprice",0) - (snap.get("c") or 0.0),
        s_kyle_ema_H60m=snap.get("s_kyle_ema_H60m"),
        vpin=snap.get("vpin"),
        atr_over_price=(atr/price) if price else 0.0,
        donchian_width_norm=snap.get("donchian_width_norm"),
        squeeze_on=snap.get("squeeze_on"),
        s_squeeze_on_dur=snap.get("s_squeeze_on_dur"),
        s_donchian_dist_upper=snap.get("s_donchian_dist_upper"),
        s_donchian_dist_lower=snap.get("s_donchian_dist_lower"),
        s_donchian_mid_dev=snap.get("s_donchian_mid_dev"),
        funding_premium_z=snap.get("funding_premium_z"),
        d_oi_rate=snap.get("d_oi_rate"),
        s_oi_rate_H60m=snap.get("s_oi_rate_H60m"),
        s_oi_rate_H180m=snap.get("s_oi_rate_H180m"),
    )
    norm = {k: z(raw.get(k), *DEFAULT_SCALE.get(k,(0,1))) for k in CANDIDATES}
    # Top-K by |z|
    top = sorted(norm.items(), key=lambda kv: abs(kv[1] or 0), reverse=True)[:topk]
    # 紧凑数组：[["rsi",0.63],["macd_dif",0.41],...]
    compact = [[k, v if v is not None else 0.0] for k,v in top]
    gate_in = dict(
        spread_bp = snap.get("spread_bp") or 0.0,
        donchian_width_norm = snap.get("donchian_width_norm"),
        funding_time_to_next_min = snap.get("funding_time_to_next_min"),
    )
    return dict(compact=compact, norm_subset=dict(top), gate=gate_in)

# ========== 极简 Gate：唯一能触发 HOLD ==========
def compute_gate(g: Dict[str,Any]) -> Tuple[str, List[str]]:
    spread = g.get("spread_bp") or 0.0
    vol = g.get("donchian_width_norm") or 0.3
    t2f = g.get("funding_time_to_next_min")
    notes=[]
    if spread>=8 and vol>=0.55 and (t2f is not None and t2f<=3):
        return "HARD_VETO", ["wide_spread","high_vol","near_funding"]
    if spread>=6: notes.append("wide_spread")
    if (t2f is not None) and t2f<=20: notes.append("near_funding")
    if not notes and spread<=3 and vol<=0.25:
        return "OPEN", []
    return "SOFT", notes

# ========== 紧凑输出解析 ==========
def parse_first_json(text:str)->dict:
    s = re.sub(r"^```.*?\n|```$", "", text.strip(), flags=re.S|re.M)
    m = re.search(r"\{.*\}", s, re.S)
    if not m: raise ValueError("no json")
    return m.group(0)

# ========== 单跳提示（最省 token） ==========
COMBINED_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are a crypto perpetuals decision engine (1h–7h). 
Roles: Buller argues up; Bearer argues down; Arbiter decides.
Use ONLY provided normalized features (compact form). 
Return JSON ONLY (no HOLD unless hard_veto=true).

Output:
{{"hard_veto":false,"dir":"L|S","ps":0-1,"conf":0-1,
 "ev_b":[["feat","+|-",0-1],...], "ev_s":[["feat","+|-",0-1],...],
 "x_checks":[["feat","rule"],...],
 "rc":{{"entry":"M|L|T","slip":int,"stop":float,"rr":float,"trail":float}},
 "notes":[]}}

Rules:
- Buller & Bearer each list 3-6 strongest evidences referencing features from input (use +/- for expected direction).
- Cross-check: include 2-4 falsifiable conditions in x_checks (what would invalidate the chosen side).
- Arbiter MUST choose 'L' or 'S' (no HOLD unless hard_veto=true received).
- Prefer MARKET only when spread small and confidence high; else LIMIT/TWAP.
- Keep text minimal; JSON only."""),
    ("user",
     """Inst:{instId} TF:{tf} TS:{ts}
Gate:{gate}
Features(compact topK):{compact}""")
])

BULL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are Buller. Argue for UP using ONLY provided compact features. JSON only:
{{"side":"B","th":0-1,"ev":[["feat","+|-",0-1],...],"risks":[],"fals":[]}}
Keep it short."""),
    ("user","Inst:{instId} TF:{tf} TS:{ts}\nFeatures:{compact}")
])

BEAR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are Bearer. Argue for DOWN using ONLY provided compact features. JSON only:
{{"side":"S","th":0-1,"ev":[["feat","+|-",0-1],...],"risks":[],"fals":[]}}
Keep it short."""),
    ("user","Inst:{instId} TF:{tf} TS:{ts}\nFeatures:{compact}")
])

ARBITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are Arbiter. Given BULL and BEAR JSON + gate + features, pick 'L' or 'S' (no HOLD unless hard_veto=true).
JSON only:
{{"hard_veto":false,"dir":"L|S","ps":0-1,"conf":0-1,
 "rc":{{"entry":"M|L|T","slip":int,"stop":float,"rr":float,"trail":float}},"notes":[]}}
Keep minimal."""),
    ("user","Gate:{gate}\nBull:{bull}\nBear:{bear}\nFeatures:{compact}")
])

# ========== 端到端决策 ==========
@dataclass
class Decision:
    hard_veto: bool
    direction: str       # "BUY_LONG"/"SELL_SHORT"/"HOLD"
    target_position: float
    leverage: int
    entry: str           # MARKET/LIMIT/TWAP_3
    slip: int
    stop_mult: float
    rr_min: float
    trail_atr_mult: float
    notes: List[str]

def leverage_from_size(size: float, cap:int=10)->int:
    return max(1, min(cap, 1 + int(9*size)))

def bind_execution(gate: str, gate_notes: List[str], raw: dict, spread: float, t2f: float|None, lev_cap:int=10)->Decision:
    if raw.get("hard_veto") is True or gate=="HARD_VETO":
        return Decision(True,"HOLD",0.0,1,"LIMIT",3,1.6,1.6,0.8,["hard_veto"]+gate_notes)
    dir_ = "BUY_LONG" if raw.get("dir")=="L" else "SELL_SHORT"
    size = float(max(0.02, min(0.8, raw.get("ps",0.3))))  # 最小探测仓 2%
    conf = float(max(0.0, min(1.0, raw.get("conf",0.5))))
    # 轻微随置信与 gate 调整
    if gate=="SOFT": size *= 0.85
    if conf<0.45: size = max(0.03, size*0.75)
    lev = leverage_from_size(size, lev_cap)
    entry_code = raw.get("rc",{}).get("entry") or ("M" if (spread<=3 and conf>=0.6 and (not t2f or t2f>20)) else ("T" if (spread>3 or (t2f and t2f<=20)) else "L"))
    entry = {"M":"MARKET","L":"LIMIT","T":"TWAP_3"}[entry_code]
    slip = int(raw.get("rc",{}).get("slip") or (2 if entry=="MARKET" else (4 if entry=="TWAP_3" else 3)))
    stop = float(raw.get("rc",{}).get("stop") or (1.6 if spread>3 else 1.2))
    rr   = float(raw.get("rc",{}).get("rr") or 1.6)
    trail= float(raw.get("rc",{}).get("trail") or 0.8)
    notes = list(raw.get("notes") or [])
    return Decision(False,dir_, round(size,4), lev, entry, slip, stop, rr, trail, notes+gate_notes)

class LLMDecider:
    def __init__(self, model="gpt-4o-mini", temperature=0.3, mode="single"):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.mode = mode  # "single" or "double"

        self.single = COMBINED_PROMPT | self.llm | StrOutputParser() | RunnableLambda(lambda s: json.loads(parse_first_json(s)))
        self.bull = BULL_PROMPT | self.llm | StrOutputParser()
        self.bear = BEAR_PROMPT | self.llm | StrOutputParser()
        self.arb  = ARBITER_PROMPT | self.llm | StrOutputParser()

    def propose(self, snapshot: Dict[str,Any]) -> Decision:
        # print(snapshot)
        pack = build_feature_pack(snapshot, topk=16)
        gate, gnotes = compute_gate(pack["gate"])
        spread = pack["gate"].get("spread_bp") or 0.0
        t2f = pack["gate"].get("funding_time_to_next_min")

        if gate=="HARD_VETO":
            return Decision(True,"HOLD",0.0,1,"LIMIT",3,1.6,1.6,0.8,["hard_veto"]+gnotes)

        if self.mode=="single":
            raw = self.single.invoke({"instId":snapshot.get("instId"),
                                      "tf":snapshot.get("tf"),
                                      "ts":snapshot.get("ts"),
                                      "gate":json.dumps({"gate":gate,"notes":gnotes}, ensure_ascii=False),
                                      "compact":json.dumps(pack["compact"], ensure_ascii=False)})
            return bind_execution(gate, gnotes, raw, spread, t2f, lev_cap=10)

        # double mode
        pp = RunnableParallel(b=self.bull, s=self.bear)
        outs = pp.invoke({"instId":snapshot.get("instId"), "tf":snapshot.get("tf"), "ts":snapshot.get("ts"),
                          "compact":json.dumps(pack["compact"], ensure_ascii=False)})
        bull = outs["b"]
        bear = outs["s"]
        arb_in = {"gate":json.dumps({"gate":gate,"notes":gnotes}),
                  "bull": bull, "bear": bear,
                  "compact": json.dumps(pack["compact"], ensure_ascii=False)}
        raw = json.loads(parse_first_json(self.arb.invoke(arb_in)))
        return bind_execution(gate, gnotes, raw, spread, t2f, lev_cap=10)

# ========== 快速试跑 ==========
if __name__=="__main__":
    snap = {
        "instId":"ETH-USDT-SWAP","tf":"15m","ts":20250923093000,
        "c":2450.0,"atr":15.0,
        "rsi":58,"macd_dif":0.22,"macd_hist":0.12,
        "s_mom_slope_H60m":0.9,"s_mom_slope_H180m":0.6,"s_mom_slope_H420m":0.3,
        "ofi_5s":0.5,"s_ofi_sum_30m":0.6,"s_cvd_delta_H60m":0.4,
        "s_spread_bp_mean_H60m":0.12,"microprice":2450.6,"s_kyle_ema_H60m":0.2,"vpin":0.1,
        "donchian_width_norm":0.32,"squeeze_on":0,"s_squeeze_on_dur":0,
        "s_donchian_dist_upper":-0.3,"s_donchian_dist_lower":0.4,"s_donchian_mid_dev":0.1,
        "funding_premium_z":-0.25,"d_oi_rate":0.003,"s_oi_rate_H60m":0.15,"s_oi_rate_H180m":0.1,
        "spread_bp":2.6,"funding_time_to_next_min":50
    }
    decider = LLMDecider(mode="single")  # 或 "double"
    d = decider.decide(snap)
    print(d)
