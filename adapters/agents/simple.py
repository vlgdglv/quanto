import json, hashlib, time
from typing import Dict, Any
from domain.interfaces import AgentPort

class SimpleAgentAdapter(AgentPort):
    """
    把你现有的 the_agent.Agent 适配到 AgentPort
    你现在的 Agent 如果有不同方法签名（比如 propose(snapshot)），只改 _call 即可。
    """
    def __init__(self, agent):
        self.agent = agent

    async def propose(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        # 任选其一：如果你的 Agent 是 agent.propose(snapshot)，就直接调用：
        # decision = await self.agent.propose(snapshot)

        # 如果是基于 messages 的：
        messages = [
            {"role": "system", "content": "You are a disciplined crypto trading analyst."},
            {"role": "user", "content": json.dumps({"snapshot": snapshot}, ensure_ascii=False)}
        ]
        resp = await self.agent.analyze(messages)  # 按你的实际方法名替换
        decision = resp["decision"]  # action/target_position/...
        return {
            "decision": decision["action"],
            "target_position": decision.get("target_position", 0.0),
            "leverage": decision.get("leverage", 1.0),
            "confidence": decision.get("confidence", 0.0),
            "reasons": decision.get("reasons", []),
            "agent_fingerprint": hashlib.sha1(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()[:12],
        }
