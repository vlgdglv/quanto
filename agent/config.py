# agent/config.py
from pydantic import BaseModel, Field

class TFConfig(BaseModel):
    use: list[str] = Field(default=["4H","1H","30m","15m"])
    main_beat: str = "10m"
    micro_beat: str = "1m"

class RiskConfig(BaseModel):
    max_loss_per_trade_pct: float = 0.006
    rr_min: float = 1.6
    spread_bp_max: float = 0.03
    funding_freeze_min: int = 5

class JudgeConfig(BaseModel):
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    self_consistency_trials: int = 1
    decision_threshold: float = 0.0

class AgentConfig(BaseModel):
    tf: TFConfig = TFConfig()
    risk: RiskConfig = RiskConfig()
    judge: JudgeConfig = JudgeConfig()
    log_dir: str = "./logs/agent"
