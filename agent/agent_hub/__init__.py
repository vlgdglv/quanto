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

from .trend_agent import invoke_trend_agent
from .trigger_agent import invoke_trigger_agent
from .trigger_agent import format_position_str_for_prompt
from .entry_agent import invoke_entry_agent
from .exit_agent import invoke_exit_agent