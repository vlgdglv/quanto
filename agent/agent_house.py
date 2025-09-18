import time
from typing import Dict, Any
from agent.orchestrator import Orchestrator, ActionProposal, ChatModelLike

class LCChatModelAdapter(ChatModelLike):
    def __init__(self, lc_chat_model):
        self.model = lc_chat_model
    def invoke(self, messages, **kwargs):
        # LangChain: model.invoke(messages) / chain.invoke(...)
        resp = self.model.invoke(messages, **kwargs)
        # 统一返回 {"content": "..."} 结构
        content = getattr(resp, "content", None)
        if content is None and isinstance(resp, dict):
            content = resp.get("content")
        return {"content": content}

class Agent:
    def __init__(self, model, temperature: float = 0.0, llm_factory=None):
        """
        model: 例如 "gpt-4o" 或你本地推理端点名
        llm_factory: Callable(role:str)-> LangChain ChatModel，用于为 RRF/DDS/EPM 分别生成模型实例
        """
        if llm_factory is None:
            raise ValueError("Please provide llm_factory(role)->LC model")

        # 为三段各建一个（也可共用）
        llm_rrf = LCChatModelAdapter(llm_factory("rrf"))
        llm_dds = LCChatModelAdapter(llm_factory("dds"))
        llm_epm = LCChatModelAdapter(llm_factory("epm"))

        self.orch = Orchestrator(
            llm_rrf=llm_rrf,
            llm_dds=llm_dds,
            llm_epm=llm_epm,
            now_ts=lambda: int(time.time())
        )

    def propose(self, snapshot: Dict[str, Any]) -> ActionProposal:
        """
        与你当前 snapshot_consumer 兼容的同步方法。
        """
        decision = self.orch.run_snapshot(snapshot)
        return decision