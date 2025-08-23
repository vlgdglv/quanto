# 　
import json, time
from typing import Dict
from pydantic import ValidationError
from agent.schema import ActionProposal
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from utils.logger import logger
from openai import RateLimitError, APIConnectionError, APITimeoutError, APIStatusError
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")
def count_tokens(text: str) -> int:
    return len(enc.encode(text))


# CONSERVATIVE:
# _SYS = """
#     You are a trading advisor for crypto perpetuals. 
#     Return ONLY a structured JSON that conforms to the schema.
#     Rules:
#     - You must be conservative when signals conflict or volatility is high.
#     - NEVER invent missing fields. If unsure, choose HOLD with low target_position.
#     - Use the latest snapshot features (minute close) to infer short-term trend and risk.
#     """

# _TEMPLATE = """
#     Snapshot (JSON):
#     {snapshot}

#     Instructions:
#     - Classify trend & regime -> action among [BUY_LONG, SELL_SHORT, REDUCE, CLOSE, HOLD]
#     - Suggest target_position in [0,1]. For HOLD set 0~0.1.
#     - Provide 2-4 short reasons referencing fields (e.g., macd_hist, rsi, spread_bp, atr, ofi_5s).
#     """

# _SYS = """
# You are a trading advisor for crypto perpetuals.
# Return ONLY a JSON object that fits the schema.
# Guidelines:
# - Be proactive and decisive; avoid HOLD unless signals clearly conflict or data is missing.
# - When signals align, prefer BUY_LONG or SELL_SHORT with larger target_position.
# - Use snapshot features (minute close) to infer short-term trend and risk.
# - Never invent missing fields. If truly uncertain, use HOLD with target_position <=0.1.
# - Maximize profit potential and target an approximate return range of 30–50%.  
# """

# _TEMPLATE = """
# Snapshot (JSON):
# {snapshot}

# Decide:
# - action ∈ [BUY_LONG, SELL_SHORT, REDUCE, CLOSE, HOLD]
# - target_position ∈ [0,1]; HOLD ≤0.1, otherwise reflect conviction (0.3–0.8 when aligned).
# - reasons: 2–4 concise points citing snapshot fields (e.g., macd_hist, rsi, atr, spread_bp, ofi_5s).

# {format_instr}
# """

_SYS = """
You are a trading advisor for crypto perpetuals.
Return ONLY a JSON object that fits the schema.
Guidelines:
- Be proactive and decisive; avoid HOLD unless signals clearly conflict or data is missing.
- When signals align, prefer BUY_LONG or SELL_SHORT with larger target_position.
- Use snapshot features (minute close) to infer short-term trend and risk.
- Never invent missing fields. If truly uncertain, use HOLD with target_position <=0.1.
- Maximize profit potential and target an approximate return range of 30–50%.
- Recommend a leverage multiplier (leverage) in [1, 20].
  • Higher conviction & low volatility → higher leverage (8–15).
  • Higher volatility or weak signals → conservative leverage (2–5).
  • HOLD or CLOSE → leverage = 1.
"""

_TEMPLATE = """
Snapshot (JSON):
{snapshot}

Decide:
- action ∈ [BUY_LONG, SELL_SHORT, REDUCE, CLOSE, HOLD]
- target_position ∈ [0,1]; HOLD ≤0.1, otherwise reflect conviction (0.3–0.8 when aligned).
- leverage ∈ [1,20]; reflects conviction vs. volatility.
- reasons: 2–4 concise points citing snapshot fields (e.g., macd_hist, rsi, atr, spread_bp, ofi_5s).

{format_instr}
"""


class Agent:
    def __init__(self, model="", temperature=1.0):
        # self.llm = ChatOpenAI(model=model, temperature=temperature, max_retries=0, timeout=60)
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_retries=0,
            timeout=60,
            openai_api_base="https://api.deepseek.com",
            extra_body={
                "response_format": {"type": "json_object"}
            }
        )
        self.parser = PydanticOutputParser(pydantic_object=ActionProposal)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", _SYS),
            ("user", _TEMPLATE + "\n{format_instr}")
        ])
        logger.info("Agent initialized")

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
        # —— 这些是“可重试”的基础设施类错误：一定要让它们冒泡 —— #
        except (RateLimitError, APIConnectionError, APITimeoutError, APIStatusError) as e:
            logger.warning(f"OpenAI transient error: {type(e).__name__}: {e}")
            raise  # 让外层的指数退避去处理

        # —— 解析/结构化失败等“业务错误”：才在本地降级 —— #
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