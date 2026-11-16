import logging

from config import RETRAINING_LOG_PATH
from utils.storage import ensure_directories


def get_logger(name: str = "churn_monitor") -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(RETRAINING_LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger