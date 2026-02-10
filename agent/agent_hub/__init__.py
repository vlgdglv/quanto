import math

def safe_float(v, ndigits=4):
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(fv):
        return None
    return round(fv, ndigits)