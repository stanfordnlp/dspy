
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
    logging.basicConfig(level=log_level)

    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(log_level))

set_log_level(level)
structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        ),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True
)

logger = structlog.get_logger()
