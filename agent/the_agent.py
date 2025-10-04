# 　
import json, time, asyncio
from typing import Dict, Optional, Any
from pydantic import ValidationError
from agent.schema import ActionProposal
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from utils.logger import logger
from openai import RateLimitError, APIConnectionError, APITimeoutError, APIStatusError
import tiktoken
from pathlib import Path
from agent.interaction_writer import InteractionWriter


enc = tiktoken.encoding_for_model("gpt-4o-mini")
def count_tokens(text: str) -> int:
    return len(enc.encode(text))


PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

def load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    return path.read_text(encoding="utf-8")

_SYS = load_prompt("sys_prompt.txt")
_TEMPLATE = load_prompt("user_template.txt")




class Agent:
    def __init__(self, 
                 model="gpt-4o", 
                 temperature=1.0,
                 interaction_writer: Optional[InteractionWriter] = None):
        self.llm = ChatOpenAI(model=model, temperature=temperature, max_retries=0, timeout=60)
        self.parser = PydanticOutputParser(pydantic_object=ActionProposal)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYS),
            ("user", _TEMPLATE + "\n{format_instr}")
        ])
        self.interactions = interaction_writer
        logger.info("Agent initialized")

    def _build_user_text(self, snapshot: Dict[str, Any]) -> str:
        return (_TEMPLATE).replace("{snapshot}", json.dumps(snapshot, ensure_ascii=False))

    def propose(self, snapshot: Dict) -> ActionProposal:
        try:
            chain = self.prompt | self.llm | self.parser
            logger.info("Prompt send, total tokens: {}."
                        .format(count_tokens(_SYS) + count_tokens(_TEMPLATE) + count_tokens(json.dumps(snapshot))))
            out: ActionProposal = chain.invoke({
                "snapshot": json.dumps(snapshot),
                "format_instr": self.parser.get_format_instructions()
            })
            logger.info("Agent Invoked.")
            return out
        except (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError) as e:
            logger.warning(f"OpenAI transient error: {type(e).__name__}: {e}")
            raise  # 让外层的指数退避去处理

        # —— 解析/结构化失败等“业务错误”：在本地降级 —— #
        except ValidationError as e:
            logger.error(f"Pydantic parse error: {e}")
            ts = snapshot.get("ts") or int(time.time() * 1000)
            return ActionProposal(
                instId=snapshot.get("instId", ""),
                tf=snapshot.get("tf", "1m"),
                ts_decision=ts,
                action="HOLD",
                target_position=0.0,
                confidence=0.2,
                reasons=[f"fallback: ValidationError"]
            )
        # —— 其他未知异常：建议也抛出，让外层处理 —— #
        except Exception as e:
            logger.exception(f"Unexpected error in propose: {e}")
            raise