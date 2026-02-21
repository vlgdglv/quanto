# agent/llm_factory.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import yaml
from functools import lru_cache
from typing import Literal, Optional, Dict, Any

from pydantic import BaseModel, Field

# LangChain chat model interfaces
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain.globals import set_debug
# set_debug(True)

from utils.config import load_cfg
from utils.logger import logger

from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.openai/.env"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CONFIG_PATH = os.getenv("AGENT_CONFIG", "configs/agent_config.yaml")

TaskType = Literal["trend", "trigger", "trading", "default"]


class LLMTaskConfig(BaseModel):
    provider: Literal["openai"] = Field(default="openai")
    model: str = Field(default="gpt-5")
    temperature: float = Field(default=0.2)
    timeout: Optional[float] = Field(default=60.0)
    max_retries: int = Field(default=2)


class LLMFactoryConfig(BaseModel):
    default: LLMTaskConfig = Field(default_factory=LLMTaskConfig)
    trend: Optional[LLMTaskConfig] = None
    trigger: Optional[LLMTaskConfig] = None
    trading: Optional[LLMTaskConfig] = None

    @staticmethod
    def from_yaml(cfg: Dict[str, Any]) -> "LLMFactoryConfig":
        llm_data = cfg.get("llm", {})

        def parse_task(name: str) -> Optional[LLMTaskConfig]:
            if name not in llm_data:
                return None
            return LLMTaskConfig(**llm_data[name])

        return LLMFactoryConfig(
            default=parse_task("default") or LLMTaskConfig(),
            trend=parse_task("trend"),
            trigger=parse_task("trigger"),
            trading=parse_task("trading"),
        )
        
    @staticmethod
    def from_env() -> "LLMFactoryConfig":
        def _tc(prefix: str, fallback_model: str) -> LLMTaskConfig:
            return LLMTaskConfig(
                provider="openai",
                model=os.getenv(f"{prefix}_MODEL", fallback_model),
                temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0.2")),
                timeout=float(os.getenv(f"{prefix}_TIMEOUT", "60")),
                max_retries=int(os.getenv(f"{prefix}_MAX_RETRIES", "2")),
            )

        default = _tc("LLM_DEFAULT", os.getenv("LLM_MODEL", "o4-mini"))
        trend = _tc("LLM_RD", os.getenv("LLM_RD_MODEL", default.model))
        trigger = _tc("LLM_TIMING", os.getenv("LLM_TIMING_MODEL", default.model))
        trading = _tc("LLM_TRADING", os.getenv("LLM_TRADING_MODEL", default.model))
        return LLMFactoryConfig(default=default, trend=trend, trigger=trigger, trading=trading)


class LLMFactory:
    def __init__(self, llm_cfg: Optional[LLMFactoryConfig] = None):
        self.cfg = llm_cfg or LLMFactoryConfig.from_env()

    def _task_cfg(self, task: TaskType) -> LLMTaskConfig:
        if task == "trend" and self.cfg.trend:
            return self.cfg.trend
        if task == "trigger" and self.cfg.trigger:
            return self.cfg.trigger
        if task == "trading" and self.cfg.trading:
            return self.cfg.trading
        return self.cfg.default

    @lru_cache(maxsize=32)
    def get_chat_model(self, task: TaskType = "default") -> BaseChatModel:
        task_cfg = self._task_cfg(task)
        if task_cfg.provider == "openai":
            return ChatOpenAI(
                model=task_cfg.model,
                temperature=task_cfg.temperature,
                timeout=task_cfg.timeout,
                max_retries=task_cfg.max_retries,
            )
        raise ValueError(f"Unsupported provider: {task_cfg.provider}")


_factory_singleton: Optional[LLMFactory] = None


def llm_factory(config_path: Optional[str] = None) -> LLMFactory:
    global _factory_singleton
    if _factory_singleton is not None:
        return _factory_singleton

    path = config_path
    if path and os.path.exists(path):
        logger.info(f"Load agents from config: {path}")
        cfg = LLMFactoryConfig.from_yaml(load_cfg(path))
    else:
        logger.info(f"Load agents from environment.")
        cfg = LLMFactoryConfig.from_env()

    _factory_singleton = LLMFactory(cfg)
    return _factory_singleton


def get_chat_model(task: TaskType = "default") -> BaseChatModel:
    return llm_factory(CONFIG_PATH).get_chat_model(task)

class PrintLLMCalls(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("[LLM START]", serialized.get("name") or serialized.get("id"))
        for i, p in enumerate(prompts, 1):
            print(f"--- Prompt {i} ---\n{p}\n")

    def on_llm_end(self, response: LLMResult, **kwargs):
        print("[LLM END] generations:", len(response.generations))
        print("[LLM TOKEN USAGE]", getattr(response, "llm_output", {}))

    def on_llm_error(self, error, **kwargs):
        print("[LLM ERROR]", error)

cb = [PrintLLMCalls()]