from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, level="INFO", enqueue=True,
           format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | {level} | {message}")
