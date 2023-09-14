import logging
import sys
from typing import Optional

APP_LOGGER_NAME = "DSP"


def get_logger(
    logger_name: str = APP_LOGGER_NAME,
    file_name: Optional[str] = None,
    logging_level: int = logging.INFO,
):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(sh)
    if file_name:
        fh = logging.FileHandler(file_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


logger = get_logger()
