# utils/config.py
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

def load_cfg():
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

    with open(Path(__file__).resolve().parents[1] / "config.yaml", "r", encoding="utf-8") as f:
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

    return resolve_env(raw_cfg)

def load_cfg_simple():
    with open(Path(__file__).resolve().parents[1] / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
