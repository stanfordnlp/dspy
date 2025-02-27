import logging
import logging.config
import sys

LOGGING_LINE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOGGING_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S"


class DSPyLoggingStream:
    """
    A Python stream for use with event logging APIs throughout DSPy (`eprint()`,
    `logger.info()`, etc.). This stream wraps `sys.stderr`, forwarding `write()` and
    `flush()` calls to the stream referred to by `sys.stderr` at the time of the call.
    It also provides capabilities for disabling the stream to silence event logs.
    """

    def __init__(self):
        self._enabled = True

    def write(self, text):
        if self._enabled:
            sys.stderr.write(text)

    def flush(self):
        if self._enabled:
            sys.stderr.flush()

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value


DSPY_LOGGING_STREAM = DSPyLoggingStream()


def disable_logging():
    """
    Disables the `DSPyLoggingStream` used by event logging APIs throughout DSPy
    (`eprint()`, `logger.info()`, etc), silencing all subsequent event logs.
    """
    DSPY_LOGGING_STREAM.enabled = False


def enable_logging():
    """
    Enables the `DSPyLoggingStream` used by event logging APIs throughout DSPy
    (`eprint()`, `logger.info()`, etc), emitting all subsequent event logs. This
    reverses the effects of `disable_logging()`.
    """
    DSPY_LOGGING_STREAM.enabled = True


def configure_dspy_loggers(root_module_name):
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "dspy_formatter": {
                    "format": LOGGING_LINE_FORMAT,
                    "datefmt": LOGGING_DATETIME_FORMAT,
                },
            },
            "handlers": {
                "dspy_handler": {
                    "formatter": "dspy_formatter",
                    "class": "logging.StreamHandler",
                    "stream": DSPY_LOGGING_STREAM,
                },
            },
            "loggers": {
                root_module_name: {
                    "handlers": ["dspy_handler"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )
