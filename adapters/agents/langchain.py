from typing import Dict, Any, Optional
from domain.interfaces import AgentPort

class LangChainAgentAdapter(AgentPort):
    """
    为 LangChain 准备的适配器：
    - 接 AgentExecutor（.invoke/.run）或 Runnable（.invoke）
    - 这里先放占位实现，等你把 chain 对外契约定下来后，填充 prompt/IO 解析即可。
    """
    def __init__(self, chain, input_key: str = "snapshot", output_key: Optional[str] = None):
        self.chain = chain
        self.input_key = input_key
        self.output_key = output_key

    async def propose(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Runnable 支持 ainvoke；AgentExecutor 可用 async run
        # out = await self.chain.ainvoke({self.input_key: snapshot})
        # 如果是同步接口，放到线程池：
        import asyncio
        out = await asyncio.to_thread(self.chain.invoke, {self.input_key: snapshot})
        data = out if self.output_key is None else out[self.output_key]
        # 2) 解析结构：要求下游 LangChain 输出统一的 dict 格式（或在此做解析）
        return {
            "decision": data["action"],
            "target_position": data.get("target_position", 0.0),
            "leverage": data.get("leverage", 1.0),
            "confidence": data.get("confidence", 0.0),
            "reasons": data.get("reasons", []),
        }
