import json, argparse

# ---- 基础：递归删除名为 snapshot 的键 ----
def strip_snapshot(x):
    if isinstance(x, dict):
        return {k: strip_snapshot(v) for k, v in x.items() if k != "snapshot"}
    if isinstance(x, list):
        return [strip_snapshot(v) for v in x]
    return x

# ---- 在 snapshot 中尽量提取 OHLC 或价格特征 ----
CAND_KEYS = {"ohlc", "bars", "candles", "klines", "kline", "k"}

def _walk(obj):
    """递归生成器"""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)

def _looks_like_dict_ohlc(arr):
    ok = 0
    for it in arr:
        if isinstance(it, dict) and all(k in it for k in ("o","h","l","c")):
            ok += 1
            if ok >= 3:
                return True
    return False

def _looks_like_list_ohlc(arr):
    ok = 0
    for it in arr:
        if isinstance(it, list) and len(it) >= 4:
            head = it[:6]
            if all((isinstance(v,(int,float)) or v is None) for v in head if v is not None):
                ok += 1
                if ok >= 3:
                    return True
    return False

def _flatten_dict_ohlc(arr):
    schema = ["ts","o","h","l","c","v"]
    out = []
    for it in arr:
        if not (isinstance(it, dict) and all(k in it for k in ("o","h","l","c"))):
            continue
        ts = it.get("ts", it.get("t"))
        v  = it.get("v", it.get("vol", it.get("volume")))
        out.append([ts, it["o"], it["h"], it["l"], it["c"], v])
    return schema, out

def _flatten_list_ohlc(arr):
    schema = ["ts","o","h","l","c","v"]
    out = []
    for it in arr:
        if not (isinstance(it, list) and len(it) >= 4):
            continue
        if len(it) >= 6:
            ts,o,h,l,c,v = it[0],it[1],it[2],it[3],it[4],it[5]
        elif len(it) == 5:
            ts,o,h,l,c,v = it[0],it[1],it[2],it[3],it[4],None
        else:  # len==4
            ts,o,h,l,c,v = None,it[0],it[1],it[2],it[3],None
        out.append([ts,o,h,l,c,v])
    return schema, out

def extract_from_snapshot(snap):
    """
    返回 (ohlc_schema, ohlc_data, price_features)
    ohlc_data 为 [[ts,o,h,l,c,v], ...]；若无 OHLC，则返回 None。
    price_features 为 dict（仅当无 OHLC 时才使用）。
    """
    if not isinstance(snap, (dict, list)):
        return None, None, {}

    best_schema, best_data = None, None

    # 1) 找 OHLC 候选
    for node in _walk(snap):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in CAND_KEYS and isinstance(v, list) and len(v) >= 3:
                    if _looks_like_dict_ohlc(v):
                        sch, dat = _flatten_dict_ohlc(v)
                    elif _looks_like_list_ohlc(v):
                        sch, dat = _flatten_list_ohlc(v)
                    else:
                        continue
                    if dat and (best_data is None or len(dat) > len(best_data)):
                        best_schema, best_data = sch, dat

        elif isinstance(node, list):
            # 直接就是数组数组
            v = node
            if _looks_like_dict_ohlc(v):
                sch, dat = _flatten_dict_ohlc(v)
            elif _looks_like_list_ohlc(v):
                sch, dat = _flatten_list_ohlc(v)
            else:
                sch, dat = None, None
            if dat and (best_data is None or len(dat) > len(best_data)):
                best_schema, best_data = sch, dat

    if best_data:
        return best_schema, best_data, {}

    # 2) 若没有 OHLC，则保留一些价格特征（尽量简单）
    pf = {}
    # last_price 尽量找一次
    for node in _walk(snap):
        if isinstance(node, dict) and "last_price" in node:
            pf["last_price"] = node["last_price"]
            break
    # trend_momentum 里的常用方向因子
    for node in _walk(snap):
        if isinstance(node, dict) and "trend_momentum" in node and isinstance(node["trend_momentum"], dict):
            tm = node["trend_momentum"]
            for k in ("ema_fast","ema_slow","macd_dif","macd_hist","rsi"):
                if k in tm:
                    pf[k] = tm[k]
            break

    return None, None, pf

def main():
    ap = argparse.ArgumentParser(description="Prune JSONL (drop 'snapshot') but keep any OHLC found inside snapshot.")
    ap.add_argument("--input", required=True, help="input JSONL")
    ap.add_argument("--output", required=True, help="output JSONL")

    args = ap.parse_args()
    INPUT_PATH = args.input
    OUTPUT_PATH = args.output
    with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except:
                continue

            # 在删除前，从 snapshot 中提取
            snap = obj.get("snapshot")
            ohlc_schema, ohlc_data, price_features = extract_from_snapshot(snap)

            # 删除所有层级的 snapshot
            pruned = strip_snapshot(obj)

            # 附加简化后的信息
            if ohlc_data:
                pruned["ohlc_schema"] = ohlc_schema
                pruned["ohlc"] = ohlc_data  # 压缩到列表，最省 token
            elif price_features:
                pruned["price_features"] = price_features

            json.dump(pruned, fout, ensure_ascii=False, separators=(",", ":"))
            fout.write("\n")

if __name__ == "__main__":
    main()
