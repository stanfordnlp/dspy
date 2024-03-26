
import logging
import os
import sys
import typing as t

import structlog

logger = structlog.get_logger()

class LogSettings:
    def __init__(self, level: str, output_type: str, method: str, file_name: t.Optional[str]) -> None:
        self.level = level
        self.output_type = output_type
        self.method = method
        self.file_name = file_name
        self._configure_structlog()

    def _configure_structlog(self):

        if self.output_type == "str":
            renderer = structlog.dev.ConsoleRenderer()
        else:
            renderer = structlog.processors.JSONRenderer()

        structlog.configure(
            processors=[
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.CallsiteParameterAdder(
                    {
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.LINENO,
                    },
                ),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                renderer,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
        )

    def set_log_level(self, level: str) -> None:
        """Set the logging level."""

        level = level.upper()
        if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("log level provider ({level}) is not one of DEBUG, INFO, WARNING, ERROR, CRITICAL")

        self.level = level

        log_level = getattr(logging, level)
        logger.setLevel(log_level)

    def set_log_output(self, method: t.Optional[str] = None, file_name: t.Optional[str] = None, output_type: t.Optional[str] = None):
        
        if method is not None and method not in ["console", "file"]:
            raise ValueError("method provided can only be 'console', 'file'")

        if method == "file" and file_name is None:
            raise ValueError("file_name must be provided when method = 'file'")

        if method is not None:
            self.method = method
            self.file_name = file_name

        if output_type is not None and output_type not in ["str", "json"]:
            raise ValueError("output_type provided can only be 'str', 'json'")

        if output_type is not None:
            self.output_type = output_type

        # Update Renderer
        self._configure_structlog()

        # Grab the root logger
        log = logging.getLogger()
        for handler in log.handlers[:]:
            log.removeHandler(handler)

        # Add new Handler
        if self.method == "file":
            assert self.file_name is not None
            log.addHandler(logging.FileHandler(self.file_name))
        else:
            log.addHandler(logging.StreamHandler(sys.stdout))


level = os.environ.get("log_level", "info").upper()

# Set Defaults
logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=level,
)

settings = LogSettings(level=level, output_type="str", method="console", file_name=None)
set_log_level = settings.set_log_level
set_log_output = settings.set_log_output
