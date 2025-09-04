# tests/conftest.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio
import logging
import yaml
import pytest
import pytest_asyncio
from infra.http_client import HttpClient

@pytest.fixture(scope="session")
def event_loop():
    # 使用独立的 event loop，避免 Windows 上的警告
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_cfg():
    def load_cfg():
        with open(Path(__file__).resolve().parents[1] / "config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return load_cfg()


@pytest_asyncio.fixture
async def http_client(test_cfg):
    """
    以异步上下文管理 HttpClient，测试中自动清理 session。
    """
    logger = logging.getLogger("HttpClientTest")
    async with HttpClient(test_cfg, logger=logger) as client:
        yield client
