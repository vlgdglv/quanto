# utils/logger.py
from loguru import logger
import os
import sys
from datetime import datetime
from pathlib import Path

ENABLE_FILE_LOG = os.getenv("ENABLE_FILE_LOG", "0") == "1"

log_dir = Path(__file__).resolve().parents[1] / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"run_{start_time}.log"

def init_logger():
    logger.remove()

    logger.add(
        sys.stdout,
        level="INFO",
        enqueue=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level} | {message}",
    )

    if ENABLE_FILE_LOG:
        logger.add(
            log_file,
            level="DEBUG",
            rotation="100 MB",
            retention="90 days",
            enqueue=True,
            encoding="utf-8",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )
        logger.info(f"File logging ENABLED → {log_file}")
    else:
        logger.info("File logging DISABLED (stdout only)")

init_logger()