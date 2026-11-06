"""Static contracts, checked by ty rather than executed by pytest.

Unused-ignore errors make the deliberately invalid calls fail if type safety is lost.
"""

from typing_extensions import assert_type

from dspy.utils.logging_utils import (
    DSPY_LOGGING_STREAM,
    DSPyLoggingStream,
    configure_dspy_loggers,
    disable_logging,
    enable_logging,
)


def check_logging_stream() -> None:
    stream = DSPyLoggingStream()
    assert_type(stream, DSPyLoggingStream)
    assert_type(stream.enabled, bool)
    assert_type(stream.write("note\n"), None)
    assert_type(stream.flush(), None)

    stream.enabled = False
    assert_type(stream.enabled, bool)

    stream.write(1)  # ty: ignore[invalid-argument-type]
    stream.enabled = "off"  # ty: ignore[invalid-assignment]
    stream.write("note\n", "extra")  # ty: ignore[too-many-positional-arguments]


def check_logging_module_api() -> None:
    assert_type(DSPY_LOGGING_STREAM, DSPyLoggingStream)
    assert_type(disable_logging(), None)
    assert_type(enable_logging(), None)
    assert_type(configure_dspy_loggers("dspy"), None)

    configure_dspy_loggers(123)  # ty: ignore[invalid-argument-type]
    configure_dspy_loggers()  # ty: ignore[missing-argument]
