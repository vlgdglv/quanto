# agents/base.py
from typing import List, Type, Any, Dict
from pydantic import BaseModel, field_validator
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from agent.agent_hub.llm_factory import get_chat_model 


class BaseAgentOutput(BaseModel):
    @field_validator("*", mode="before") 
    @classmethod
    def _coerce_list(cls, v, info):
        # 这是一个通用技巧：只针对定义为 List 的字段生效
        field_type = info.annotation
        # 简单判断是否是 List 类型 (Python 3.8+ 可能需要 get_origin)
        if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
            if v is None: return []
            if isinstance(v, str): return [v.strip()]
            if isinstance(v, (list, tuple)): return [str(x).strip() for x in v if x is not None]
        return v
    
    
def create_agent_chain(
    output_model: Type[BaseModel],
    prompt_text: str,
    model_name: str = "gpt-4o",
) -> Runnable:
    
    parser = PydanticOutputParser(pydantic_object=output_model)
    
    prompt = ChatPromptTemplate.from_template(
        prompt_text,
        partial_variables={"fomrat_instructions": parser.get_format_instructions()},
    )
    
    llm = get_chat_model(task=model_name)

    # 构造 Chain
    chain = prompt | llm | parser
    return chain