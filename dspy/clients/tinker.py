"""Tinker-hosted models behind DSPy's engine contract.

Tinker samples tokens. The tinker_cookbook renderer for the model turns the
request's messages and tools into a token prompt and parses the sampled tokens
back into text, thinking and tool calls. The Tinker SDK owns the connection,
its authentication and its own transport retries; DSPy's LM layer owns the
response cache, retries, candidates, history and callbacks.
"""

import asyncio
import importlib
import json
import threading
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from dspy._vendor.lm15.result import apply_client_side_stop
from dspy.clients.base_lm import BaseLM
from dspy.clients.engines.base import validate_request
from dspy.lm15 import (
    Adaptation,
    AuthError,
    BillingError,
    FunctionTool,
    InvalidRequestError,
    Message,
    ProviderError,
    RateLimitError,
    Request,
    Response,
    ServerError,
    TextPart,
    ThinkingPart,
    TimeoutError,
    ToolCallPart,
    ToolResultPart,
    TransportError,
    UnsupportedFeatureError,
    UnsupportedModelError,
    Usage,
)
from dspy.utils.lazy_import import require

_FEATURE = "dspy.TinkerLM"
_PROVIDER = "tinker"
# Generation settings the sampler and the renderer have no way to honour.
_UNSUPPORTED_CONFIG = (
    "response_format",
    "reasoning",
    "cache",
    "frequency_penalty",
    "presence_penalty",
    "service_tier",
    "user_id",
    "store",
    "logprobs",
    "probabilities",
    "extensions",
)


def _sdk() -> tuple[Any, Any, Any]:
    """Import the Tinker SDK and cookbook; a missing package raises the `dspy[tinker]` install hint."""
    tinker = require("tinker", extra="tinker", feature=_FEATURE)
    cookbook = require("tinker_cookbook", extra="tinker", feature=_FEATURE)
    # Reading an attribute imports a lazy module, or raises the hint for a missing one.
    _params, _version = tinker.SamplingParams, cookbook.__version__
    return (
        tinker,
        importlib.import_module("tinker_cookbook.renderers"),
        importlib.import_module("tinker_cookbook.model_info"),
    )


def _unsupported(message: str, feature: str) -> UnsupportedFeatureError:
    return UnsupportedFeatureError(message, feature=feature, provider=_PROVIDER)


@contextmanager
def _sdk_errors(tinker):
    """Translate the SDK's expected failures; anything else keeps its own type."""
    try:
        yield
    except tinker.RequestFailedError as exc:
        error = {"server": ServerError, "user": InvalidRequestError}.get(str(exc.category), ProviderError)
        raise error(exc.message, provider=_PROVIDER, request_id=exc.request_id) from exc
    except (tinker.AuthenticationError, tinker.PermissionDeniedError) as exc:
        raise AuthError(
            exc.message,
            provider=_PROVIDER,
            status=exc.status_code,
            env_keys=("TINKER_API_KEY",),
            credential_hint="Pass api_key= to TinkerLM, set TINKER_API_KEY, or run `tinker auth login`",
        ) from exc
    except tinker.APITimeoutError as exc:
        raise TimeoutError(exc.message, provider=_PROVIDER) from exc
    except tinker.APIConnectionError as exc:
        raise TransportError(exc.message, provider=_PROVIDER) from exc
    except tinker.APIStatusError as exc:
        error = ProviderError
        for source, target in (
            (tinker.RateLimitError, RateLimitError),
            (tinker.BillingError, BillingError),
            (tinker.NotFoundError, UnsupportedModelError),
            ((tinker.BadRequestError, tinker.UnprocessableEntityError, tinker.ConflictError), InvalidRequestError),
            (tinker.InternalServerError, ServerError),
        ):
            if isinstance(exc, source):
                error = target
                break
        raise error(exc.message, provider=_PROVIDER, status=exc.status_code) from exc


def _text(parts, where: str) -> str:
    texts: list[str] = []
    for part in parts:
        if not isinstance(part, TextPart):
            raise _unsupported(f"Tinker renders text only; {where} contains a {part.type} part", part.type)
        texts.append(part.text)
    return "".join(texts)


def _check_config(config) -> None:
    unsupported = [name for name in _UNSUPPORTED_CONFIG if getattr(config, name) not in (None, {})]
    if unsupported:
        raise _unsupported(
            f"Tinker sampling cannot honour {unsupported}; the renderer decides the prompt format "
            "and thinking, and the sampler takes only temperature, max_tokens, top_p, top_k, stop and seed",
            unsupported[0],
        )


def _offered_tools(request: Request) -> tuple[list[FunctionTool], list[Adaptation]]:
    """The tools to render, as narrowed by tool_choice, and what sampling cannot enforce."""
    tools = []
    for tool in request.tools:
        if not isinstance(tool, FunctionTool):
            raise _unsupported(f"Tinker renders function tools only, not the {tool.name} builtin", "tools")
        tools.append(tool)
    choice = request.config.tool_choice
    if choice is None:
        return tools, []
    if choice.mode == "none":
        return [], []
    if choice.allowed:
        tools = [tool for tool in tools if tool.name in choice.allowed]
    notes: list[Adaptation] = []
    if choice.mode == "required":
        notes.append(
            Adaptation(
                field="config.tool_choice",
                action="substituted",
                asked="required",
                applied="auto",
                reason="Tinker samples freely and cannot force a tool call; only the allowed tools are rendered.",
            )
        )
    if choice.parallel is False:
        notes.append(
            Adaptation(
                field="config.tool_choice.parallel",
                action="dropped",
                asked=False,
                reason="Tinker sampling cannot limit the model to one tool call per turn.",
            )
        )
    return tools, notes


def _renderer_messages(request: Request, renderers) -> list[dict[str, Any]]:
    """The conversation as renderer messages, without the system prompt."""
    messages = []
    tool_names = {}
    for message in request.messages:
        if message.role == "user":
            messages.append({"role": "user", "content": _text(message.parts, "a user message")})
        elif message.role == "assistant":
            content, calls = [], []
            for part in message.parts:
                if isinstance(part, TextPart):
                    content.append({"type": "text", "text": part.text})
                elif isinstance(part, ThinkingPart):
                    content.append({"type": "thinking", "thinking": part.text})
                elif isinstance(part, ToolCallPart):
                    tool_names[part.id] = part.name
                    body = renderers.ToolCall.FunctionBody(name=part.name, arguments=json.dumps(part.input))
                    calls.append(renderers.ToolCall(id=part.id, function=body))
                else:
                    raise _unsupported(f"Tinker renders no {part.type} parts in an assistant message", part.type)
            if not any(item["type"] == "thinking" for item in content):
                content = "".join(item["text"] for item in content)
            rendered = {"role": "assistant", "content": content}
            if calls:
                rendered["tool_calls"] = calls
            messages.append(rendered)
        elif message.role == "tool":
            for part in message.parts:
                if not isinstance(part, ToolResultPart):
                    raise _unsupported(f"Tinker renders no {part.type} parts in a tool message", part.type)
                rendered = {"role": "tool", "content": _text(part.content, "a tool result"), "tool_call_id": part.id}
                name = part.name or tool_names.get(part.id)
                if name is not None:
                    rendered["name"] = name
                messages.append(rendered)
        else:
            raise _unsupported(f"Tinker chat templates have no {message.role} role", message.role)
    return messages


class TinkerEngine:
    """Samples one response from a Tinker base model or LoRA checkpoint.

    The sampling client, tokenizer and renderer are created on first use,
    once, and shared by an `AsyncTinkerEngine` wrapping this engine.
    """

    supports_function_calling = True
    supported_params = frozenset({"max_tokens", "temperature", "top_p", "top_k", "stop", "seed"})

    def __init__(
        self,
        model: str | None = None,
        *,
        model_path: str | None = None,
        renderer: str | None = None,
        api_key: str | None = None,
        sampling_client: Any = None,
    ) -> None:
        """Configure the engine; nothing connects to Tinker until the first request.

        Args:
            model: The Tinker base model, such as `"Qwen/Qwen3-8B"`. Optional
                with `model_path` or `sampling_client`, which know their base model.
            model_path: A `tinker://` sampler path to LoRA weights, which takes
                precedence over `model` for sampling.
            renderer: A tinker_cookbook renderer name. Defaults to the one
                recommended for the base model.
            api_key: A Tinker API key. Defaults to the SDK's own lookup
                (`TINKER_API_KEY`, then a stored login). Never saved.
            sampling_client: An existing `tinker.SamplingClient` to sample with.
        """
        if model is None and model_path is None and sampling_client is None:
            raise ValueError("TinkerEngine needs a base model, a tinker:// model_path, or a sampling_client")
        self.model = model
        self.model_path = model_path
        self.renderer = renderer
        self._api_key = api_key
        self._sampling_client = sampling_client
        self._client = sampling_client
        self._renderer = None
        self._lock = threading.Lock()

    def _session(self) -> tuple[Any, Any]:
        """The sampling client and renderer, created once."""
        with self._lock:
            if self._renderer is None:
                tinker, renderers, model_info = _sdk()
                with _sdk_errors(tinker):
                    if self._client is None:
                        service = tinker.ServiceClient(api_key=self._api_key)
                        self._client = service.create_sampling_client(
                            model_path=self.model_path,
                            base_model=self.model,
                        ).result()
                    base_model = self.model or self._client.get_base_model()
                    name = self.renderer or model_info.get_recommended_renderer_name(base_model)
                    self._renderer = renderers.get_renderer(name, self._client.get_tokenizer(), model_name=base_model)
            return self._client, self._renderer

    def _translate(
        self, request: Request
    ) -> tuple[list[dict[str, Any]], str | None, list[dict[str, Any]], list[Adaptation]]:
        """Check the request and translate it for the renderer, before any I/O."""
        validate_request(request)
        _check_config(request.config)
        messages = _renderer_messages(request, _sdk()[1])
        system = (
            None
            if request.system is None
            else (request.system if isinstance(request.system, str) else _text(request.system, "the system prompt"))
        )
        tools, notes = _offered_tools(request)
        specs = [
            {"name": tool.name, "description": tool.description or "", "parameters": tool.parameters} for tool in tools
        ]
        return messages, system, specs, notes

    def _prepare(self, request: Request, translated, renderer) -> tuple[Any, Any]:
        """The token prompt and sampling parameters for one candidate."""
        messages, system, specs, _ = translated
        if specs:
            try:
                prefix = renderer.create_conversation_prefix_with_tools(specs, system or "")
            except NotImplementedError:
                raise _unsupported(f"The {type(renderer).__name__} Tinker renderer has no tool-calling format", "tools")
            messages = [*prefix, *messages]
        elif system is not None:
            messages = [{"role": "system", "content": system}, *messages]
        config = request.config
        options = {name: getattr(config, name) for name in ("max_tokens", "temperature", "top_p", "top_k", "seed")}
        params = _sdk()[0].SamplingParams(
            stop=renderer.get_stop_sequences(),
            **{name: value for name, value in options.items() if value is not None},
        )
        return renderer.build_generation_prompt(messages), params

    def _response(self, request: Request, renderer, prompt, sample, notes) -> Response:
        sequence = sample.sequences[0]
        tokens = list(sequence.tokens)
        parsed, _termination = renderer.parse_response(tokens)
        content = parsed["content"]
        parts = []
        for item in [{"type": "text", "text": content}] if isinstance(content, str) else content:
            if item["type"] == "thinking":
                parts.append(ThinkingPart(item["thinking"]))
            elif item["type"] == "text":
                parts.append(TextPart(item["text"]))
        # A call the renderer could not parse stays visible as the text the model wrote.
        parts.extend(TextPart(call.raw_text) for call in parsed.get("unparsed_tool_calls", ()))
        for index, call in enumerate(parsed.get("tool_calls", ())):
            parts.append(
                ToolCallPart(
                    id=call.id or f"call_{index}", name=call.function.name, input=json.loads(call.function.arguments)
                )
            )
        if not parts:
            parts.append(TextPart(""))
        if sequence.stop_reason == "length":
            finish_reason = "length"
        elif any(isinstance(part, ToolCallPart) for part in parts):
            finish_reason = "tool_call"
        else:
            finish_reason = "stop"
        response = Response(
            id=sequence.sequence_id,
            model=request.model,
            message=Message.assistant(parts),
            finish_reason=finish_reason,
            usage=Usage(input_tokens=prompt.length, output_tokens=len(tokens)),
        )
        stop = request.config.stop
        if stop:
            response = apply_client_side_stop(response, stop)
            notes = [
                *notes,
                Adaptation(
                    field="config.stop",
                    action="client_side",
                    asked=list(stop),
                    applied=list(stop),
                    reason="The sampler stops at the end of the assistant turn; DSPy trims the parsed reply at these "
                    "sequences. Usage counts every sampled token.",
                ),
            ]
        return replace(response, adaptations=(*response.adaptations, *notes)) if notes else response

    def complete(self, request: Request) -> Response:
        translated = self._translate(request)
        client, renderer = self._session()
        prompt, params = self._prepare(request, translated, renderer)
        with _sdk_errors(_sdk()[0]):
            sample = client.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
        return self._response(request, renderer, prompt, sample, translated[3])

    def dump_state(self) -> dict[str, Any]:
        return {"model": self.model, "model_path": self.model_path, "renderer": self.renderer}

    @classmethod
    def load_state(cls, state: dict[str, Any]) -> "TinkerEngine":
        return cls(**state)

    def __deepcopy__(self, memo) -> "TinkerEngine":
        # Copied programs share the connection, as LM.copy() shares engines.
        return self

    def __getstate__(self) -> dict[str, Any]:
        # A client created here carries the API key in its own pickle, so only a passed-in one is kept.
        state = dict(self.__dict__, _api_key=None, _client=self._sampling_client, _renderer=None)
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()


class AsyncTinkerEngine:
    """The async counterpart of a `TinkerEngine`, sharing its sampling client and renderer.

    The Tinker SDK runs requests on its own event loop thread, so one client
    serves synchronous callers and any number of event loops.
    """

    supports_function_calling = TinkerEngine.supports_function_calling
    supported_params = TinkerEngine.supported_params

    def __init__(self, engine: TinkerEngine) -> None:
        self.engine = engine

    async def complete(self, request: Request) -> Response:
        engine = self.engine
        translated = engine._translate(request)
        if engine._renderer is None:
            # The first call creates the client and loads the tokenizer, both blocking.
            await asyncio.to_thread(engine._session)
        client, renderer = engine._session()
        prompt, params = engine._prepare(request, translated, renderer)
        with _sdk_errors(_sdk()[0]):
            sample = await client.sample(prompt=prompt, num_samples=1, sampling_params=params)
        return engine._response(request, renderer, prompt, sample, translated[3])

    def dump_state(self) -> dict[str, Any]:
        return self.engine.dump_state()

    @classmethod
    def load_state(cls, state: dict[str, Any]) -> "AsyncTinkerEngine":
        return cls(TinkerEngine.load_state(state))


class TinkerLM(BaseLM):
    """A DSPy LM that samples from a Tinker base model or LoRA checkpoint.

    Messages and tools are rendered with the tinker_cookbook renderer for the
    base model, and sampled tokens are parsed back into text, thinking and
    tool calls. Requires `pip install "dspy[tinker]"`.

    Example:
        ```python
        lm = dspy.TinkerLM("Qwen/Qwen3-8B", temperature=0.7, max_tokens=1024)
        tuned = dspy.TinkerLM(model_path="tinker://<run-id>/sampler_weights/final")
        ```
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        model_path: str | None = None,
        renderer: str | None = None,
        api_key: str | None = None,
        sampling_client: Any = None,
        **kwargs: Any,
    ) -> None:
        """Create a Tinker LM; nothing connects to Tinker until the first call.

        Args:
            model: The Tinker base model, such as `"Qwen/Qwen3-8B"`.
            model_path: A `tinker://` sampler path to LoRA weights. The base
                model is read from the checkpoint when `model` is omitted.
            renderer: A tinker_cookbook renderer name, such as
                `"qwen3_disable_thinking"`. Defaults to the one recommended
                for the base model.
            api_key: A Tinker API key. Defaults to `TINKER_API_KEY` or a
                stored `tinker auth login`. Never saved.
            sampling_client: An existing `tinker.SamplingClient` to sample with.
                Without `model_path`, `cache` defaults to `False`, because the
                cache key would name only the base model.
            **kwargs: `BaseLM` options: `temperature`, `max_tokens`, `cache`,
                `num_retries`, `callbacks`, and the generation defaults
                `top_p`, `top_k`, `stop` and `seed`.
        """
        if sampling_client is not None and model_path is None:
            # The cache key cannot name the weights behind a bare sampling client.
            kwargs.setdefault("cache", False)
            model = model or sampling_client.get_base_model()
        engine = TinkerEngine(
            model, model_path=model_path, renderer=renderer, api_key=api_key, sampling_client=sampling_client
        )
        # The response cache keys on this string, so it names the weights and any renderer override.
        label = f"tinker/{model_path or model}" + (f"?renderer={renderer}" if renderer else "")
        super().__init__(model=label, engine=engine, async_engine=AsyncTinkerEngine(engine), **kwargs)

    def dump_state(self) -> dict[str, Any]:
        """Return the LM options and the engine's model, model_path and renderer, for `TinkerLM(**state)`."""
        state = super().dump_state()
        engine = state.pop("engine")
        state.pop("async_engine")
        state.update(engine["state"])
        return state
