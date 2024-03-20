
import logging
import os

import structlog

level = os.environ.get("LOG_LEVEL", "INFO").upper()

def set_log_level(level: str) -> None:
    """Set the logging level."""
    level = level.upper()
    if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        raise ValueError("log level provider ({level}) is not one of DEBUG, INFO, WARNING, ERROR, CRITICAL")

    log_level = getattr(logging, level)

    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(log_level))

set_log_level(level)
logger = structlog.get_logger()
