"""The optimizer-facing execution trace.

When tracing is active (``dspy.settings.trace`` is a list), every predictor
invocation appends one ``(predictor, inputs, prediction)`` tuple. Teleprompters
(GEPA, MIPROv2, SIMBA, bootstrapping) mine these tuples to attribute examples
and feedback to individual predictors. This tuple shape is a public contract:
custom predictors that execute outside ``dspy.Predict`` (e.g. harness- or
backend-based modules) should call :func:`record_trace` so optimizers can see
their invocations.
"""

from typing import Any, Mapping

from dspy.dsp.utils.settings import settings


def record_trace(predictor: Any, inputs: Mapping[str, Any], prediction: Any) -> None:
    """Append one ``(predictor, inputs, prediction)`` tuple to the active trace.

    No-op when tracing is disabled. Enforces ``settings.max_trace_size`` as a
    bounded FIFO. ``inputs`` is copied into a plain dict so later mutation of
    the caller's kwargs cannot corrupt the trace.
    """
    if settings.trace is None or settings.max_trace_size <= 0:
        return
    trace = settings.trace
    if len(trace) >= settings.max_trace_size:
        trace.pop(0)
    trace.append((predictor, dict(inputs), prediction))
