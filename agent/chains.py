# agents/base.py
from typing import List, Type, Any, Dict
from pydantic import BaseModel, field_validator
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from agent.llm_factory import get_chat_model 


class BaseAgentOutput(BaseModel):
    @field_validator("*", mode="before") 
    @classmethod
    def _coerce_list(cls, v, info):
        field_type = info.annotation
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
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    llm = get_chat_model(task=model_name)

    chain = prompt | llm | parser
    return chain