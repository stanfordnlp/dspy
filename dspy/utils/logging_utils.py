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
        """Initialize the stream in an enabled state."""
        self._enabled = True

    def write(self, text):
        """Write text to the current ``sys.stderr`` stream when logging is enabled.

        Args:
            text: Text emitted by the logging subsystem.
        """
        if self._enabled:
            sys.stderr.write(text)

    def flush(self):
        """Flush the current ``sys.stderr`` stream when logging is enabled."""
        if self._enabled:
            sys.stderr.flush()

    @property
    def enabled(self):
        """Whether writes and flushes should be forwarded to ``sys.stderr``."""
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        """Enable or disable forwarding to ``sys.stderr``.

        Args:
            value: ``True`` to emit log output, ``False`` to suppress it.
        """
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
    """Configure DSPy's root logger to emit through ``DSPY_LOGGING_STREAM``.

    This helper installs a single named stream handler, formats log lines using
    DSPy's standard timestamped format, and prevents duplicate propagation to
    ancestor loggers.

    Args:
        root_module_name: Root logger name to configure, for example ``"dspy"``.
    """
    formatter = logging.Formatter(fmt=LOGGING_LINE_FORMAT, datefmt=LOGGING_DATETIME_FORMAT)

    dspy_handler_name = "dspy_handler"
    handler = logging.StreamHandler(stream=DSPY_LOGGING_STREAM)
    handler.setFormatter(formatter)
    handler.set_name(dspy_handler_name)

    logger = logging.getLogger(root_module_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for existing_handler in logger.handlers[:]:
        if getattr(existing_handler, "name", None) == dspy_handler_name:
            logger.removeHandler(existing_handler)

    logger.addHandler(handler)
