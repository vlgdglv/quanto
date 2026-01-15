# agent/agent_hub/definitions.py
from typing import List
from pydantic import Field
from agent.agent_hub.base import BaseAgentOutput, create_agent_chain
from agent.agent_hub.prompts import RD_PROMPT_TEMPLATE, TIMING_PROMPT_TEMPLATE


class RDOutput(BaseAgentOutput): # 继承 Base，自动获得 list 清洗能力
    regime: str
    regime_confidence: float
    direction: str
    direction_confidence: float
    reasons: List[str] = []
    invalidation: List[str] = []


rd_agent_chain = create_agent_chain(RDOutput, RD_PROMPT_TEMPLATE, model_name="rd_model")