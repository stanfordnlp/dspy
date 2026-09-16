import copy as copy_module
import importlib
import inspect
from typing import Any, TextIO

from dspy.dsp.utils import settings
from dspy.lm15 import Request, Response
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.inspect_history import pretty_print_history

MAX_HISTORY_SIZE = 10_000
GLOBAL_HISTORY = []
LM_CLASS_STATE_KEY = "_dspy_lm_class"
_BUILTIN_LM_CLASS_PATH = "dspy.clients.lm.LM"
_MIGRATION_GUIDE = "https://dspy.ai/community/normalized-lm-api-migration/"


def _import_lm_class(class_path: str) -> type:
    parts = class_path.split(".")
    last_error = None

    for split_index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_index])
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                last_error = exc
                continue
            raise

        try:
            for attr in parts[split_index:]:
                obj = getattr(obj, attr)
        except AttributeError as exc:
            last_error = exc
            continue

        if not isinstance(obj, type):
            raise TypeError(f"Serialized LM class `{class_path}` did not resolve to a class.")
        return obj

    raise ImportError(f"Could not import serialized LM class `{class_path}`.") from last_error


def _check_engine(engine, async_engine):
    if engine is not None and not callable(getattr(engine, "complete", None)):
        raise TypeError("An engine must implement complete(Request) -> Response")
    if async_engine is not None and not callable(getattr(async_engine, "complete", None)):
        raise TypeError("An async engine must implement `async complete(Request) -> Response`")


class BaseLM:
    """Base class for DSPy language models.

    Every LM executes through an *engine*: an object implementing
    `complete(Request) -> Response`, optionally `stream(Request)` yielding lm15
    stream events, plus an async counterpart with `async complete` and an async
    `stream`. `dspy.LM` selects its built-in engines from the model string; any
    other LM supplies engines directly. DSPy's LM layer owns response caching,
    retries, candidate fan-out, usage accounting, callbacks and history around
    that single-request contract, so engines never add a second layer of those.

    Calls speak lm15 objects: `lm(request)` returns one `dspy.lm15.Response`,
    `lm.generate(request, n=...)` returns several, and `lm("text")` is a
    convenience that renders one user message and returns a list of outputs.
    OpenAI-style `messages=` dictionaries and `forward()` implementations were
    removed in DSPy 3.5; see the migration guide at
    https://dspy.ai/community/normalized-lm-api-migration/.

    Persistent custom state belongs in dump_state/load_state. Runtime clients
    are shared by copy(), while DSPy history, callbacks and kwargs are isolated.
    """

    def __init__(
        self,
        model,
        model_type="chat",
        temperature=None,
        max_tokens=None,
        cache=True,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,
        engine=None,
        async_engine=None,
        **kwargs,
    ):
        """Initialize a base language model.

        Args:
            model: The model identifier.
            model_type: The LM API type, such as `"chat"`, `"text"`, or
                `"responses"`.
            temperature: The default sampling temperature.
            max_tokens: The default maximum number of output tokens.
            cache: Whether requests should use DSPy's cache by default.
            callbacks: Optional instance-level callback handlers.
            num_retries: The default number of engine attempts after a
                transient failure.
            engine: An object implementing `complete(Request) -> Response`.
                Engines are borrowed; DSPy never closes them.
            async_engine: The async counterpart used by `acall`/`agenerate`.
            **kwargs: Additional default generation options stored in
                `self.kwargs`, in the OpenAI vocabulary (`top_p`, `stop`,
                `reasoning_effort`, ...).
        """
        _check_engine(engine, async_engine)
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.callbacks = list(callbacks or [])
        self.num_retries = num_retries
        self.kwargs = self._get_initial_kwargs(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []
        self._engine_spec = engine
        self._async_engine_spec = async_engine

    def _get_initial_kwargs(self, *, temperature, max_tokens, **kwargs) -> dict[str, Any]:
        return dict(temperature=temperature, max_tokens=max_tokens, **kwargs)

    @property
    def engine(self):
        """The configured engine (an object, or a named selection on `dspy.LM`)."""
        return self._engine_spec

    @property
    def async_engine(self):
        """The configured async engine object, if any."""
        return self._async_engine_spec

    @property
    def supports_function_calling(self) -> bool:
        """Whether the model supports function calling (tool use)."""
        return bool(getattr(self._engine_spec, "supports_function_calling", False))

    @property
    def supports_reasoning(self) -> bool:
        """Whether the model supports native reasoning (extended thinking)."""
        return bool(getattr(self._engine_spec, "supports_reasoning", False))

    @property
    def supports_response_schema(self) -> bool:
        """Whether the model supports structured output via response schema."""
        return bool(getattr(self._engine_spec, "supports_response_schema", False))

    @property
    def supported_params(self) -> set[str]:
        """Set of supported OpenAI-style generation option names for the model."""
        return set(getattr(self._engine_spec, "supported_params", ()) or ())

    @with_callbacks
    def __call__(self, prompt=None, *, cache=None, rollout_id=None, **options):
        """Run one model call.

        Args:
            prompt: A `dspy.lm15.Request`, or a string rendered as one user
                message with this LM's generation defaults.
            cache: Override this LM's response-cache setting for the call.
            rollout_id: Distinguish otherwise identical cached requests.
            **options: Generation options for a string prompt, in the OpenAI
                vocabulary (`temperature`, `max_tokens`, `n`, ...), plus
                client settings (`api_key`, `api_base`, ...) for this call.
                A `Request` accepts only the client settings; its `config`
                already holds the generation options.

        Returns:
            The `Response` for a `Request`; a list of outputs for a string.
        """
        from dspy.clients.execution import execute, finalize, prepare

        call = prepare(self, prompt, options, cache=cache, rollout_id=rollout_id)
        return finalize(self, call, execute(self, call))

    @with_callbacks
    async def acall(self, prompt=None, *, cache=None, rollout_id=None, **options):
        """Async equivalent of `__call__`, with the same execution ownership."""
        import asyncio

        from dspy.clients.execution import aexecute, finalize, prepare

        # Canonical media snapshots may read local files. Context variables
        # propagate to the worker; callbacks/finalization stay on the caller.
        call = await asyncio.to_thread(prepare, self, prompt, options, cache=cache, rollout_id=rollout_id,
                                       asynchronous=True)
        return finalize(self, call, await aexecute(self, call))

    @with_callbacks
    def generate(self, request: Request, *, n: int = 1, cache=None, rollout_id=None, **client_options) -> list[Response]:
        """Return `n` responses for one request, as DSPy adapters do.

        Candidates are separate engine requests, cached and recorded together
        as one call. Each response contains exactly one assistant message.
        `client_options` are per-call client settings such as `api_base`.
        """
        from dspy.clients.execution import execute, finalize, prepare

        call = prepare(self, request, client_options, cache=cache, rollout_id=rollout_id, n=n, candidates=True)
        return finalize(self, call, execute(self, call))

    @with_callbacks
    async def agenerate(self, request: Request, *, n: int = 1, cache=None, rollout_id=None, **client_options) -> list[Response]:
        """Async equivalent of `generate`."""
        import asyncio

        from dspy.clients.execution import aexecute, finalize, prepare

        call = await asyncio.to_thread(prepare, self, request, client_options, cache=cache, rollout_id=rollout_id, n=n,
                                       candidates=True, asynchronous=True)
        return finalize(self, call, await aexecute(self, call))

    def dump_state(self) -> dict[str, Any]:
        """Return a sanitized reconstruction state for this LM.

        Subclasses whose state is captured by `BaseLM.__init__` can use this
        default. Subclasses with extra persistent state should override both
        `dump_state` and `load_state`.

        Returns:
            A dictionary that can be passed to `BaseLM.load_state`. The state
            excludes API keys.
        """
        filtered_kwargs = {key: value for key, value in self.kwargs.items() if key not in ("api_key", LM_CLASS_STATE_KEY)}
        return {
            LM_CLASS_STATE_KEY: f"{type(self).__module__}.{type(self).__qualname__}",
            "model": self.model,
            "model_type": self.model_type,
            "cache": self.cache,
            "num_retries": getattr(self, "num_retries", 3),
            **filtered_kwargs,
        }

    @classmethod
    def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False) -> "BaseLM":
        """Reconstruct an LM from `dump_state` output.

        Legacy states without a class marker load as `dspy.LM`. Custom LM
        classes must be importable by their module-qualified class path and are
        only loaded when `allow_custom_lm_class=True`.

        Args:
            state: Serialized LM state produced by `dump_state`.
            allow_custom_lm_class: If True, allow importing and loading custom
                `BaseLM` subclasses recorded in `state`. Enable only for trusted
                state.

        Returns:
            The reconstructed LM instance.

        Raises:
            ValueError: If `state` references a custom LM class and
                `allow_custom_lm_class` is False.
            ImportError: If the serialized LM class cannot be imported.
            TypeError: If the serialized class is not a `BaseLM` subclass.
        """
        state = dict(state)
        class_path = state.pop(LM_CLASS_STATE_KEY, None)

        if cls is BaseLM:
            if class_path is None:
                # Legacy saved programs did not record the concrete LM class.
                from dspy.clients.lm import LM

                return LM(**state)

            if class_path != _BUILTIN_LM_CLASS_PATH and not allow_custom_lm_class:
                raise ValueError(
                    f"Refusing to import custom serialized LM class `{class_path}`. "
                    "Pass allow_unsafe_lm_state=True when loading trusted files to enable custom LM classes."
                )

            lm_cls = _import_lm_class(class_path)
            if not issubclass(lm_cls, BaseLM):
                raise TypeError(f"Serialized LM class `{class_path}` must be a subclass of dspy.BaseLM.")
            if "allow_custom_lm_class" in inspect.signature(lm_cls.load_state).parameters:
                return lm_cls.load_state(state, allow_custom_lm_class=allow_custom_lm_class)
            return lm_cls.load_state(state)

        return cls(**state)

    def copy(self, **kwargs):
        """Return a copy of the language model with updated parameters.

        The default implementation makes a shallow runtime copy. Engines,
        provider clients, sessions, and local model handles are preserved by
        reference. DSPy-owned mutable state is isolated for `history`, the
        `callbacks` list, and the `kwargs` dict. Other attributes are shared by
        reference. Subclasses with additional mutable DSPy-owned state should
        override this method.

        Args:
            **kwargs: Attribute or generation-option updates to apply to the
                copy. For example, `lm.copy(rollout_id=1, temperature=1.0)`
                returns an LM whose requests use a different rollout ID at
                non-zero temperature to bypass cache collisions.

        Returns:
            A copied LM instance.
        """

        new_instance = copy_module.copy(self)
        new_instance.history = []
        new_instance.callbacks = list(getattr(self, "callbacks", []) or [])
        new_instance.kwargs = dict(getattr(self, "kwargs", {}) or {})

        for key, value in kwargs.items():
            if hasattr(new_instance, key):
                setattr(new_instance, key, value)
            if (key in new_instance.kwargs) or (not hasattr(self, key)):
                if value is None:
                    new_instance.kwargs.pop(key, None)
                else:
                    new_instance.kwargs[key] = value
        if hasattr(new_instance, "_warned_zero_temp_rollout"):
            new_instance._warned_zero_temp_rollout = False

        return new_instance

    def inspect_history(self, n: int = 1, file: "TextIO | None" = None) -> None:
        pretty_print_history(self.history, n, file=file)

    def update_history(self, entry):
        if settings.disable_history:
            return

        # Global LM history
        if len(GLOBAL_HISTORY) >= MAX_HISTORY_SIZE:
            GLOBAL_HISTORY.pop(0)

        GLOBAL_HISTORY.append(entry)

        if settings.max_history_size == 0:
            return

        # dspy.LM.history
        if len(self.history) >= settings.max_history_size:
            self.history.pop(0)

        self.history.append(entry)

        # Per-module history
        caller_modules = settings.caller_modules or []
        for module in caller_modules:
            if len(module.history) >= settings.max_history_size:
                module.history.pop(0)
            module.history.append(entry)


def inspect_history(n: int = 1, file: "TextIO | None" = None) -> None:
    """The global history shared across all LMs.

    Args:
        n: Number of recent entries to display. Defaults to 1.
        file: An optional file-like object to write output to. When
            provided, ANSI color codes are automatically disabled.
            Defaults to `None` (prints to stdout).
    """
    pretty_print_history(GLOBAL_HISTORY, n, file=file)
