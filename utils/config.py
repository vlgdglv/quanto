# utils/config.py
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

from utils.logger import logger

def load_cfg(cfg_path: str | None = None):
    
    base_dir = Path(__file__).resolve().parents[1]

    cfg_file = Path(cfg_path) if cfg_path else (base_dir / "config.yaml")

    load_dotenv(base_dir / ".env")

    with open(cfg_file, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    def resolve_env(obj):
        if isinstance(obj, dict):
            return {k: resolve_env(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve_env(v) for v in obj]
        if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            varname = obj[2:-1]
            return os.getenv(varname, "")
        return obj

    cfg = resolve_env(raw_cfg)

    def contains_live_mode(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == "mode" and str(v).lower() == "live":
                    return True
                if contains_live_mode(v):
                    return True
        elif isinstance(obj, list):
            return any(contains_live_mode(v) for v in obj)
        return False

    if contains_live_mode(cfg):
        logger.warning("⚠️ Live mode detected!!! Use with extreme caution!")

    return cfg

def load_cfg_simple():
    with open(Path(__file__).resolve().parents[1] / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
