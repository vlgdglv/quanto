# utils/logger.py
from loguru import logger
import sys
from datetime import datetime
from pathlib import Path

log_dir = Path(__file__).resolve().parents[1] / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"run_{start_time}.log"

logger.remove()

logger.add(
    sys.stdout,
    level="INFO",
    enqueue=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level} | {message}",
)

logger.add(
    log_file,
    level="DEBUG",
    rotation="100 MB",  
    retention="90 days",
    enqueue=True,
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

logger.info(f"Logger initialized. Writing logs to {log_file}")