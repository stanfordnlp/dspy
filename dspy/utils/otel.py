"""OpenTelemetry tracing for DSPy programs, built on the callback system.

LM and tool spans follow the OpenTelemetry GenAI semantic conventions
(https://opentelemetry.io/docs/specs/semconv/gen-ai/). Spans for DSPy-specific
components without a GenAI equivalent (modules, adapters, evaluation) carry
``dspy.*``-namespaced attributes.

Requires the ``opentelemetry-api`` package (``pip install dspy[otel]``). Spans
are emitted through whichever tracer provider is passed in (or the globally
registered one), so traces can be exported to any OTLP-compatible backend.
"""

import json
import os
from typing import Any

from pydantic import BaseModel

from dspy.core.types import LMMessage, LMRequest, LMResponse
from dspy.dsp.utils.settings import settings
from dspy.utils.callback import BaseCallback

_CAPTURE_CONTENT_ENV_VAR = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
_SUPPORTED_SEMCONV_VERSIONS = (1,)


def _json_default(value: Any) -> Any:
    from dspy.primitives.example import Example

    if isinstance(value, Example):
        return value.toDict()
    if isinstance(value, BaseModel):
        return value.model_dump()
    return repr(value)


def _serialize(value: Any) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=False)


def _to_genai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-shaped chat messages to the GenAI semconv message structure."""
    normalized = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts = [{"type": "text", "content": content}]
        elif isinstance(content, list):
            parts = content
        else:
            parts = []
        normalized.append({"role": message.get("role", "user"), "parts": parts})
    return normalized


def _lm_input_messages(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract OpenAI-shaped chat messages from `BaseLM.__call__` inputs."""
    if inputs.get("messages") is not None:
        return inputs["messages"]
    if inputs.get("prompt") is not None:
        return [{"role": "user", "content": inputs["prompt"]}]

    items = inputs.get("items") or ()
    request = inputs.get("request")
    if request is None and items and isinstance(items[0], LMRequest):
        request = items[0]
        items = items[1:]
    if request is not None:
        return [message.model_dump(exclude_none=True) for message in request.messages]

    messages = []
    for item in items:
        if isinstance(item, LMMessage):
            messages.append(item.model_dump(exclude_none=True))
        elif isinstance(item, str):
            messages.append({"role": "user", "content": item})
        else:
            messages.append({"role": "user", "content": _serialize(item)})
    return messages


def _lm_output_messages(outputs: LMResponse | list[dict[str, Any] | str]) -> list[dict[str, Any]]:
    """Convert LM outputs (typed or legacy) to the GenAI semconv message structure."""
    if isinstance(outputs, LMResponse):
        parts = [{"type": "text", "content": outputs.text}] if outputs.text else []
        return [{"role": "assistant", "parts": parts}]

    messages = []
    for output in outputs:
        if isinstance(output, str):
            parts = [{"type": "text", "content": output}]
        elif output.get("text"):
            parts = [{"type": "text", "content": output["text"]}]
        else:
            parts = []
        messages.append({"role": "assistant", "parts": parts})
    return messages


def _usage_attributes(outputs: Any) -> dict[str, int]:
    """Extract gen_ai.usage.* attributes; only typed `LMResponse` outputs carry usage."""
    if not isinstance(outputs, LMResponse):
        return {}
    usage = outputs.usage_as_dict()
    attributes = {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    if input_tokens is not None:
        attributes["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens is not None:
        attributes["gen_ai.usage.output_tokens"] = output_tokens
    return attributes


class OtelCallback(BaseCallback):
    """A callback that emits OpenTelemetry spans for DSPy component calls.

    Spans nest through the OpenTelemetry context, so DSPy spans parent
    correctly under any surrounding application spans and vice versa.

    Args:
        tracer_provider: The tracer provider to emit spans through. Defaults
            to the globally registered provider, which is a no-op unless an
            OpenTelemetry SDK has been configured.
        capture_content: Whether to record prompts, completions, tool
            arguments/results, and module/adapter inputs/outputs on spans.
            Defaults to True unless the standard
            ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` environment
            variable is set to a false value (``false`` or ``0``).
        semconv_version: The DSPy GenAI attribute-layout version. Only version
            1 exists today; the parameter lets attribute layouts evolve with
            the (still-in-development) GenAI semantic conventions without
            breaking existing dashboards.

    Example:

    ```
    import dspy

    dspy.configure(callbacks=[dspy.OtelCallback()])
    ```
    """

    def __init__(
        self,
        tracer_provider: Any | None = None,
        capture_content: bool | None = None,
        semconv_version: int = 1,
    ):
        try:
            from opentelemetry import context as otel_context
            from opentelemetry import trace as otel_trace
        except ImportError as e:
            raise ImportError(
                "OtelCallback requires the `opentelemetry-api` package. "
                "Install it with `pip install dspy[otel]`."
            ) from e

        if semconv_version not in _SUPPORTED_SEMCONV_VERSIONS:
            raise ValueError(
                f"Unsupported semconv_version {semconv_version!r}; supported versions: {_SUPPORTED_SEMCONV_VERSIONS}."
            )

        from dspy.__metadata__ import __version__

        self._otel_trace = otel_trace
        self._otel_context = otel_context
        provider = tracer_provider if tracer_provider is not None else otel_trace.get_tracer_provider()
        self._tracer = provider.get_tracer("dspy", __version__)

        if capture_content is None:
            env_value = os.environ.get(_CAPTURE_CONTENT_ENV_VAR)
            capture_content = env_value is None or env_value.strip().lower() not in ("false", "0")
        self.capture_content = capture_content
        self.semconv_version = semconv_version

        # call_id -> (span, context attach token); starts and ends are paired
        # by with_callbacks, and both run in the same thread/task context.
        self._active_spans: dict[str, tuple[Any, Any]] = {}

    def _start_span(self, call_id: str, name: str, kind: Any, attributes: dict[str, Any]):
        attributes = {key: value for key, value in attributes.items() if value is not None}
        span = self._tracer.start_span(name, kind=kind, attributes=attributes)
        token = self._otel_context.attach(self._otel_trace.set_span_in_context(span))
        self._active_spans[call_id] = (span, token)

    def _end_span(self, call_id: str, exception: Exception | None, attributes: dict[str, Any] | None = None):
        span, token = self._active_spans.pop(call_id)
        self._otel_context.detach(token)
        if attributes:
            span.set_attributes({key: value for key, value in attributes.items() if value is not None})
        if exception is not None:
            span.record_exception(exception)
            span.set_status(self._otel_trace.Status(self._otel_trace.StatusCode.ERROR, str(exception)))
        span.end()

    def on_module_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        from dspy.predict.predict import Predict

        attributes = {"dspy.module.type": type(instance).__name__}
        if isinstance(instance, Predict):
            attributes["dspy.module.signature"] = instance.signature.signature
        if self.capture_content:
            attributes["dspy.module.input"] = _serialize(inputs)
        self._start_span(call_id, type(instance).__name__, self._otel_trace.SpanKind.INTERNAL, attributes)

    def on_module_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None):
        attributes = {}
        if self.capture_content and outputs is not None:
            attributes["dspy.module.output"] = _serialize(outputs)
        self._end_span(call_id, exception, attributes)

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        model = instance.model
        if "/" in model:
            provider, model_name = model.split("/", 1)
        else:
            provider, model_name = None, model

        attributes = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model_name,
            "gen_ai.request.temperature": instance.kwargs.get("temperature"),
            "gen_ai.request.max_tokens": instance.kwargs.get("max_tokens"),
        }
        if self.capture_content:
            attributes["gen_ai.input.messages"] = _serialize(_to_genai_messages(_lm_input_messages(inputs)))
        self._start_span(call_id, f"chat {model_name}", self._otel_trace.SpanKind.CLIENT, attributes)

    def on_lm_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        attributes = {}
        if outputs is not None:
            attributes.update(_usage_attributes(outputs))
            if isinstance(outputs, LMResponse) and outputs.model is not None:
                attributes["gen_ai.response.model"] = outputs.model
            if self.capture_content:
                attributes["gen_ai.output.messages"] = _serialize(_lm_output_messages(outputs))
        self._end_span(call_id, exception, attributes)

    def on_adapter_format_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self._start_adapter_span(call_id, instance, inputs, method="format")

    def on_adapter_format_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        self._end_adapter_span(call_id, outputs, exception)

    def on_adapter_parse_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        self._start_adapter_span(call_id, instance, inputs, method="parse")

    def on_adapter_parse_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        self._end_adapter_span(call_id, outputs, exception)

    def _start_adapter_span(self, call_id: str, instance: Any, inputs: dict[str, Any], method: str):
        attributes = {"dspy.adapter.type": type(instance).__name__, "dspy.adapter.method": method}
        if self.capture_content:
            attributes["dspy.adapter.input"] = _serialize(inputs)
        self._start_span(
            call_id, f"{type(instance).__name__}.{method}", self._otel_trace.SpanKind.INTERNAL, attributes
        )

    def _end_adapter_span(self, call_id: str, outputs: Any | None, exception: Exception | None):
        attributes = {}
        if self.capture_content and outputs is not None:
            attributes["dspy.adapter.output"] = _serialize(outputs)
        self._end_span(call_id, exception, attributes)

    def on_tool_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        attributes = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": instance.name,
            "gen_ai.tool.description": instance.desc,
            "gen_ai.tool.type": "function",
        }
        if self.capture_content:
            attributes["gen_ai.tool.call.arguments"] = _serialize(inputs["kwargs"])
        self._start_span(
            call_id, f"execute_tool {instance.name}", self._otel_trace.SpanKind.INTERNAL, attributes
        )

    def on_tool_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        attributes = {}
        if self.capture_content and outputs is not None:
            attributes["gen_ai.tool.call.result"] = _serialize(outputs)
        self._end_span(call_id, exception, attributes)

    def on_evaluate_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        attributes = {"dspy.evaluate.num_examples": len(instance.devset)}
        self._start_span(call_id, "Evaluate", self._otel_trace.SpanKind.INTERNAL, attributes)

    def on_evaluate_end(self, call_id: str, outputs: Any | None, exception: Exception | None = None):
        from dspy.evaluate.evaluate import EvaluationResult

        attributes = {}
        if isinstance(outputs, EvaluationResult):
            attributes["dspy.evaluate.score"] = float(outputs.score)
        self._end_span(call_id, exception, attributes)


def enable_otel_tracing(
    tracer_provider: Any | None = None,
    capture_content: bool | None = None,
    semconv_version: int = 1,
) -> OtelCallback:
    """Enable OpenTelemetry tracing for all DSPy component calls.

    Registers an `OtelCallback` as a global DSPy callback. Calling this more
    than once returns the already-registered callback instead of adding a
    duplicate.

    Args:
        tracer_provider: The tracer provider to emit spans through. Defaults
            to the globally registered provider.
        capture_content: Whether to record prompts, completions, tool
            arguments/results, and module/adapter inputs/outputs on spans. See
            `OtelCallback` for the default behavior.
        semconv_version: The DSPy GenAI attribute-layout version.

    Returns:
        The registered `OtelCallback`.

    Example:

    ```
    import dspy

    dspy.enable_otel_tracing()
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    dspy.Predict("question -> answer")(question="What is DSPy?")
    ```
    """
    existing_callbacks = settings.get("callbacks", [])
    for callback in existing_callbacks:
        if isinstance(callback, OtelCallback):
            return callback

    callback = OtelCallback(
        tracer_provider=tracer_provider,
        capture_content=capture_content,
        semconv_version=semconv_version,
    )
    settings.configure(callbacks=[*existing_callbacks, callback])
    return callback
