# agent/chat_models.py
from typing import Dict, Any,List
from langchain_openai import ChatOpenAI

def llm_factory(role: str):
    # 你可以按角色设不同温度/模型/超参
    if role == "rrf":
        return ChatOpenAI(model="gpt-4o", temperature=0.0)
    if role == "dds":
        return ChatOpenAI(model="gpt-4o", temperature=0.0)
    if role == "epm":
        return ChatOpenAI(model="gpt-4o", temperature=0.0)
    # 默认
    return ChatOpenAI(model="gpt-4o", temperature=0.0)

class ChatModelLike:
    def __init__(self, **kwargs): ...
    def invoke(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
