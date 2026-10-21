
import logging
from pathlib import Path

LOG_FILE = Path("./application.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")

    # formatter = logging.Formatter(
    #     "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    #     datefmt="%Y-%m-%d %H:%M:%S"
    # )
    # handler.setFormatter(formatter)

    logger.addHandler(handler)