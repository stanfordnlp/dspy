"""Host side of the sandbox dspy facade.

Sandboxed code (a ``dspy.Flex`` module or a ``dspy.RLM`` REPL block) gets a stand-in ``dspy``
module, ``_facade_shim.py``, whose predictor constructors and calls are proxies. Only JSON crosses
the sandbox boundary, over the interpreter's tool-call protocol (``CodeInterpreter.tools``):

- ``__dspy_construct__``: the shim asks the host to build a real predictor
  (``FacadeInvocation.construct``) from a serialized signature/kwargs and gets a string handle.
- ``__dspy_call__``: the shim runs a predictor by handle (``FacadeInvocation.call``); the host
  makes the real LM call and returns the prediction's fields as JSON.

Callables and signatures travel as markers the host resolves: tools by name (``__dspy_tool__``)
against the host module's tools, ``dspy.Signature(...)`` results as ``__dspy_sig__`` payloads.
All per-forward state (constructed predictors, the predictor-call budget, custom-type originals,
the last LM infrastructure error) lives in a ``FacadeInvocation`` created for that forward.

Every LM call made for sandboxed code runs on a ``SandboxLM`` over the host LM, which admits only
generation parameters (``SANDBOX_LM_KWARGS``) and meters the calls.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
import secrets
import textwrap
from pathlib import Path
from typing import Any, Callable

import pydantic
from pydantic_core import PydanticSerializationError, to_jsonable_python

import dspy
from dspy.adapters.types.base_type import Type
from dspy.clients.base_lm import BaseLM
from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.signatures.signature import make_signature
from dspy.utils.exceptions import LMError

logger = logging.getLogger(__name__)

# Tool names the shim uses to call the host; registered by FacadeInvocation.install.
CONSTRUCT_TOOL = "__dspy_construct__"
CALL_TOOL = "__dspy_call__"

# Predictors the sandbox shim may construct and the host builds
BRIDGEABLE_KINDS = ("Predict", "ChainOfThought", "RLM", "CodeAct", "ProgramOfThought", "ReAct", "ReActV2")
# The shim's dspy.Signature(...) emits this marker so the host can rebuild a Signature.
SIGNATURE_MARKER = "__dspy_sig__"
# The shim passes tools by name (callables can't cross the JSON boundary); the host resolves the name
# against the host module's tools.
TOOL_MARKER = "__dspy_tool__"
# dspy.Type values cross the boundary as JSON objects tagged with their type name (see to_sandbox_value).
TYPE_MARKER = "__dspy_type__"

# The sandbox-side dspy shim, injected as text into each interpreter.
SHIM_SETUP = (Path(__file__).parent / "_facade_shim.py").read_text(encoding="utf-8")
# Shipped to the shim so it can refuse a sandbox that is the host's own memory image (host process or fork).
HOST_TOKEN_VAR = "__dspy_host_token"
# Shipped to the shim so it can add each dspy.Type's sandbox_methods to that type's sandbox class.
TYPE_METHODS_VAR = "__dspy_type_methods"
_HOST_PROCESS_TOKEN = secrets.token_hex(16)


def is_reserved_sandbox_name(name: str) -> bool:
    """Names the facade owns in the sandbox namespace: ``dspy`` and the ``_dspy``/``__dspy`` prefixes
    of its internals and host-injected variables."""
    return name == "dspy" or name.startswith(("_dspy", "__dspy"))


def _accepts_interpreter_factory(cls: type) -> bool:
    """True if ``cls.__init__`` takes an ``interpreter_factory`` parameter."""
    try:
        return "interpreter_factory" in inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def _resolve_signature(signature: Any, custom_types: dict[str, type] | None = None) -> Any:
    """Turn a shim signature payload back into something a host predictor accepts."""
    if isinstance(signature, dict) and signature.get(SIGNATURE_MARKER):
        # marker always carries a string signature; make_signature applies instructions if given
        return make_signature(signature["signature"], signature.get("instructions"), custom_types=custom_types)
    if isinstance(signature, str):
        return make_signature(signature, custom_types=custom_types)
    return signature


def _jsonable(value: Any) -> Any:
    """Coerce a predictor output field to a JSON-serializable value."""
    try:
        return to_jsonable_python(value)
    except PydanticSerializationError:
        return value


def prediction_to_fields(pred: Any) -> dict[str, Any]:
    """Serialise a host ``dspy.Prediction`` (or dict) to a JSON-able field dict for the sandbox."""
    store = getattr(pred, "_store", None)
    if store is None:
        if isinstance(pred, dict):
            store = pred
        else:
            raise CodeInterpreterError(
                f"A bridged predictor must return a dspy.Prediction; got {type(pred).__name__}"
            )
    fields = {k: _jsonable(v) for k, v in dict(store).items()}
    try:
        json.dumps(fields)
    except TypeError as e:
        raise CodeInterpreterError(
            "A bridged predictor returned a field that cannot cross the sandbox boundary "
            f"(must be JSON-serializable): {e}"
        ) from e
    return fields


def to_sandbox_value(value: Any) -> Any:
    """Encode ``dspy.Type`` values, recursively, as JSON objects tagged with ``TYPE_MARKER``.

    The sandbox shim rebuilds them as ``_DspyValue`` dicts (with the type's ``sandbox_methods``), and
    ``from_sandbox_value`` turns them back into the host type when they cross back.
    """
    if isinstance(value, Type):
        fields = {name: to_sandbox_value(getattr(value, name)) for name in type(value).model_fields}
        return {TYPE_MARKER: _type_tag(type(value)), **fields}
    if isinstance(value, tuple) and hasattr(value, "_asdict"):
        return to_sandbox_value(value._asdict())  # named tuples cross as dicts, as the interpreters serialize them
    if isinstance(value, dict):
        return {k: to_sandbox_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_sandbox_value(v) for v in value]
    return value


def from_sandbox_value(value: Any) -> Any:
    """Rebuild ``TYPE_MARKER``-tagged JSON objects from the sandbox as ``dspy.Type`` values, recursively.

    Construction runs the type's own validation; DSPy's media types never read host files or the
    network while validating.
    """
    if isinstance(value, dict):
        name = value.get(TYPE_MARKER)
        if not isinstance(name, str):
            return {k: from_sandbox_value(v) for k, v in value.items()}
        fields = {k: from_sandbox_value(v) for k, v in value.items() if k != TYPE_MARKER}
        try:
            # Fields cross by name (see to_sandbox_value), so accept names as well as aliases.
            return _sandbox_type(name).model_validate(fields, by_name=True)
        except pydantic.ValidationError as e:
            raise CodeInterpreterError(f"Invalid {name} value from the sandbox: {e}") from e
    if isinstance(value, list):
        return [from_sandbox_value(v) for v in value]
    return value


def _type_subclasses(cls: type[Type] = Type) -> list[type[Type]]:
    found = []
    for sub in cls.__subclasses__():
        found.append(sub)
        found.extend(_type_subclasses(sub))
    return found


def _type_tag(cls: type[Type]) -> str:
    """The ``TYPE_MARKER`` value for ``cls``: its module-qualified name, so same-named types stay distinct."""
    return f"{cls.__module__}.{cls.__qualname__}"


def _sandbox_type(tag: str) -> type[Type]:
    """The loaded ``dspy.Type`` subclass with this ``TYPE_MARKER`` tag."""
    matches = [cls for cls in _type_subclasses() if _type_tag(cls) == tag]
    if not matches:
        raise CodeInterpreterError(f"No loaded dspy.Type subclass is {tag!r}.")
    # A class redefined under the same name (e.g. a re-run notebook cell) supersedes the earlier one.
    return matches[-1]


def _sandbox_type_methods() -> dict[str, str]:
    """The ``sandbox_methods`` source of each loaded ``dspy.Type`` that has any, keyed by ``TYPE_MARKER`` tag."""
    methods = {}
    for cls in _type_subclasses():
        source = textwrap.dedent(cls.sandbox_methods()).strip()
        if source:
            methods[_type_tag(cls)] = source
    return methods


def preload_sandbox_packages(interpreter: Any) -> None:
    """Ask an interpreter that loads packages at startup for the ``sandbox_packages`` of every loaded ``dspy.Type``.

    A value of any type can reach the sandbox (e.g. from a tool without a return annotation, or from a
    sub-agent), and packages cannot be added once the interpreter has started. An interpreter that cannot
    add them (a custom ``deno_command``, an already-started process) still runs; type methods that need
    the packages then fail to import.
    """
    preload = getattr(interpreter, "preload_packages", None)
    packages = list(dict.fromkeys(package for cls in _type_subclasses() for package in cls.sandbox_packages()))
    if preload is None or not packages:
        return
    try:
        preload(packages)
    except (ValueError, CodeInterpreterError) as e:
        logger.debug("Could not preload sandbox packages %s: %s", packages, e)


def restore_custom_types(value: Any, originals: dict[str, Any]) -> Any:
    """Substitute serialized custom-type strings back with the original host objects."""
    if isinstance(value, str) and value in originals:
        return originals[value]
    if isinstance(value, dict):
        return {k: restore_custom_types(v, originals) for k, v in value.items()}
    if isinstance(value, list):
        return [restore_custom_types(v, originals) for v in value]
    return value


# Generation parameters sandboxed code may set per LM call; routing, credential, and transport options are not.
SANDBOX_LM_KWARGS = frozenset({
    "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "n", "seed",
    "logprobs", "top_logprobs", "response_format",
    "reasoning", "reasoning_effort",
    "tools", "tool_choice", "parallel_tool_calls",
    "cache", "rollout_id", "prompt_cache", "prompt_cache_key",
})
_LM_CALL_INPUTS = frozenset({"prompt", "messages", "request"})


class SandboxLM(BaseLM):
    """The host LM as sandboxed code may call it: only ``SANDBOX_LM_KWARGS`` per call, and ``reserve(n)`` before
    each, one slot per requested completion."""

    def __init__(self, lm: Any, reserve: Callable[[int], None] | None = None) -> None:
        # lm may be any callable RLM accepts as sub_lm, not only a BaseLM. Adapters and Predict read the
        # generation defaults off the proxy; credentials and routing stay on the wrapped LM.
        super().__init__(model=getattr(lm, "model", None), model_type=getattr(lm, "model_type", "chat"))
        self.kwargs = {k: v for k, v in (getattr(lm, "kwargs", None) or {}).items() if k in SANDBOX_LM_KWARGS}
        self._lm = lm
        self._reserve = reserve

    @staticmethod
    def _check(kwargs: dict[str, Any], *, allow_inputs: bool = True) -> None:
        allowed = SANDBOX_LM_KWARGS | _LM_CALL_INPUTS if allow_inputs else SANDBOX_LM_KWARGS
        rejected = sorted(kwargs.keys() - allowed)
        if rejected:
            raise TypeError(
                f"Sandboxed code may not set LM option(s) {rejected}; allowed per call: {sorted(SANDBOX_LM_KWARGS)}."
            )

    def _build_adapter_request(self, prompt, options):
        from dspy.clients.requests import build_request

        # Only overrides come from sandboxed code. Host defaults remain trusted,
        # including provider-specific settings that are not exposed on this proxy.
        self._check(options or {}, allow_inputs=False)
        return build_request(self._lm, prompt, options)

    def _admit(self, kwargs: dict[str, Any], args: tuple = ()) -> None:
        self._check(kwargs)
        if self._reserve is not None:
            prompt = args[0] if args else kwargs.get("request", kwargs.get("prompt"))
            # Typed requests are singular even when the wrapped LM defaults to n > 1.
            count = 1 if isinstance(prompt, dspy.lm15.Request) else max(1, int(kwargs.get("n", self.kwargs.get("n")) or 1))
            self._reserve(count)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._admit(kwargs, args)
        return self._lm(*args, **kwargs)

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        self._admit(kwargs, args)
        return await self._lm.acall(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self._admit(kwargs)
        return self._lm.forward(*args, **kwargs)

    async def aforward(self, *args: Any, **kwargs: Any) -> Any:
        self._admit(kwargs)
        return await self._lm.aforward(*args, **kwargs)

    def copy(self, **kwargs: Any) -> SandboxLM:
        self._check(kwargs)
        return SandboxLM(self._lm.copy(**kwargs), self._reserve)

    # Adapters choose structured outputs and native tool calling by these; answer for the wrapped model.
    @property
    def supports_function_calling(self) -> bool:
        return getattr(self._lm, "supports_function_calling", False)

    @property
    def supports_reasoning(self) -> bool:
        return getattr(self._lm, "supports_reasoning", False)

    @property
    def supports_response_schema(self) -> bool:
        return getattr(self._lm, "supports_response_schema", False)

    @property
    def supported_params(self) -> set[str]:
        return getattr(self._lm, "supported_params", set())


class FacadeInvocation:
    """Per-forward host side of the sandbox dspy facade.

    Builds the predictors the shim asks for (keyed by the sandbox attribute name) from the host
    module's tools and interpreter factory, runs them by handle under a predictor-call budget on a
    ``SandboxLM``, and restores custom-type inputs. Each forward gets its own ``FacadeInvocation``.
    ``lm`` defaults to ``dspy.settings.lm`` at call time; ``reserve`` meters every LM call.
    """

    def __init__(
        self,
        tools: dict[str, Any],
        interpreter_factory: Callable[[], Any],
        max_predictor_calls: int | None,
        *,
        custom_types: dict[str, type] | None = None,
        lm: Any = None,
        reserve: Callable[[int], None] | None = None,
        originals: dict[str, Any] | None = None,
    ) -> None:
        self._tools = tools
        self._interpreter_factory = interpreter_factory
        self._max_predictor_calls = max_predictor_calls
        self._custom_types = custom_types
        self._lm = lm
        self._reserve = reserve
        self._originals = originals or {}
        self._predictors: dict[str, Any] = {}
        self._calls = 0
        self._lm_error: tuple[LMError, str] | None = None

    def install(self, interpreter: Any) -> None:
        """Register the tools the shim calls, then install the ``dspy`` facade in the sandbox."""
        interpreter.tools.update({CONSTRUCT_TOOL: self.construct, CALL_TOOL: self.call})
        interpreter.execute(
            SHIM_SETUP, variables={HOST_TOKEN_VAR: _HOST_PROCESS_TOKEN, TYPE_METHODS_VAR: _sandbox_type_methods()}
        )

    def construct(self, kind: str, signature: Any, attr_name: str, kwargs: dict[str, Any] | None = None) -> str:
        self._lm_error = None
        if kind not in BRIDGEABLE_KINDS:
            raise CodeInterpreterError(
                f"dspy.{kind} is not supported through the sandbox dspy bridge yet "
                f"(bridgeable: {', '.join(BRIDGEABLE_KINDS)})"
            )
        self._predictors[attr_name] = self._build_predictor(kind, signature, kwargs)
        return attr_name

    def call(self, handle: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        self._lm_error = None
        self._calls += 1
        budget = self._max_predictor_calls
        if budget is not None and self._calls > budget:
            raise CodeInterpreterError(
                f"Sandboxed forward exceeded its predictor-call budget ({budget}). "
                "Raise max_predictor_calls if this is expected."
            )
        predictor = self._predictors.get(handle)
        if predictor is None:
            raise CodeInterpreterError(f"Unknown predictor handle: {handle!r}")
        restored = {k: restore_custom_types(v, self._originals) for k, v in (inputs or {}).items()}
        try:
            with self._lm_scope():
                return prediction_to_fields(predictor(**restored))
        except LMError as e:
            tag = f"[dspy bridge lm-error #{self._calls}]"
            self._lm_error = (e, tag)
            raise CodeInterpreterError(f"{tag} {type(e).__name__}: {e}") from e

    def _lm_scope(self) -> contextlib.AbstractContextManager[Any]:
        """dspy.context(lm=SandboxLM) for one bridged call; a scoped override, so nested predictors use it too."""
        lm = self._lm if self._lm is not None else dspy.settings.lm
        if lm is None:
            return contextlib.nullcontext()
        if isinstance(lm, SandboxLM) and self._reserve is None:
            return dspy.context(lm=lm)
        return dspy.context(lm=SandboxLM(lm, self._reserve))

    def _build_predictor(self, kind: str, signature: Any, kwargs: dict[str, Any] | None) -> Any:
        cls = getattr(dspy, kind)
        if "interpreter_factory" in (kwargs or {}):
            raise CodeInterpreterError(
                f"dspy.{kind} built in the sandbox runs on the host module's interpreter; "
                "sandboxed code cannot choose its interpreter_factory."
            )
        extra = {k: self._decode_tools(v) for k, v in (kwargs or {}).items()}
        # A code-executing sub-predictor always runs its inner code on the host module's backend.
        if _accepts_interpreter_factory(cls):
            extra["interpreter_factory"] = self._interpreter_factory
        return cls(_resolve_signature(signature, self._custom_types), **extra)

    def _decode_tools(self, value: Any) -> Any:
        """Turn the shim's tool name-markers back into the real host tool objects."""
        if isinstance(value, dict) and TOOL_MARKER in value:
            name = value[TOOL_MARKER]
            if name not in self._tools:
                raise CodeInterpreterError(
                    f"Sandboxed code referenced tool {name!r}, which was not provided to the host module's "
                    "tools. Functions authored inside the sandbox cannot be handed to a bridged sub-predictor."
                )
            return self._tools[name]
        if isinstance(value, list):
            return [self._decode_tools(v) for v in value]
        if isinstance(value, dict):
            return {k: self._decode_tools(v) for k, v in value.items()}
        return value
