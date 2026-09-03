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
import functools
import inspect
import json
from pathlib import Path
from typing import Any, Callable

from pydantic_core import PydanticSerializationError, to_jsonable_python

import dspy
from dspy.adapters.types.base_type import Type as _CustomType
from dspy.clients.base_lm import BaseLM
from dspy.primitives.code_interpreter import SUB_DSPY_FACTORY_NAME, CodeInterpreterError
from dspy.signatures.signature import make_signature
from dspy.utils.exceptions import LMError

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
# Value of the shim's ``dspy_interpreter_factory`` global: the host substitutes its own factory.
FACTORY_MARKER = "__dspy_interpreter_factory__"

# The sandbox-side dspy shim, injected as text into each interpreter.
SHIM_SETUP = (Path(__file__).parent / "_facade_shim.py").read_text(encoding="utf-8")


def is_reserved_sandbox_name(name: str) -> bool:
    """Names the facade owns in the sandbox namespace: ``dspy``, the ``_dspy``/``__dspy`` prefixes
    of its internals and host-injected variables, and the nested-interpreter factory name."""
    return name == "dspy" or name.startswith(("_dspy", "__dspy")) or name == SUB_DSPY_FACTORY_NAME


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


def _tool_entrypoint(tool: Any) -> Callable[..., Any]:
    """A callable for the interpreter's tool registry."""
    func = getattr(tool, "func", None)
    if func is None:
        return tool

    @functools.wraps(func)
    def entrypoint(**kwargs: Any) -> Any:
        return tool(**kwargs)

    return entrypoint


def _restoring_entrypoint(fn: Callable[..., Any], originals: dict[str, Any]) -> Callable[..., Any]:
    """Wrap a tool entrypoint so serialized custom-type inputs arrive as the original objects."""

    @functools.wraps(fn)
    def entrypoint(**kwargs: Any) -> Any:
        return fn(**{k: _restore_custom_types(v, originals) for k, v in kwargs.items()})

    return entrypoint


def _collect_custom_type_originals(value: Any, out: dict[str, Any]) -> None:
    """Record custom-type instances by their serialized string, recursing into containers."""
    if isinstance(value, _CustomType):
        out[value.serialize_model()] = value
    elif isinstance(value, dict):
        for v in value.values():
            _collect_custom_type_originals(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _collect_custom_type_originals(v, out)


def _restore_custom_types(value: Any, originals: dict[str, Any]) -> Any:
    """Substitute serialized custom-type strings back with the original host objects."""
    if isinstance(value, str) and value in originals:
        return originals[value]
    if isinstance(value, dict):
        return {k: _restore_custom_types(v, originals) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_custom_types(v, originals) for v in value]
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
    """The host LM as sandboxed code may call it: only ``SANDBOX_LM_KWARGS`` per call, ``reserve(1)`` before each."""

    def __init__(self, lm: Any, reserve: Callable[[int], None] | None = None) -> None:
        # lm may be any callable RLM accepts as sub_lm, not only a BaseLM. Adapters and Predict read the
        # generation defaults off the proxy; credentials and routing stay on the wrapped LM.
        super().__init__(model=getattr(lm, "model", None), model_type=getattr(lm, "model_type", "chat"))
        self.kwargs = {k: v for k, v in (getattr(lm, "kwargs", None) or {}).items() if k in SANDBOX_LM_KWARGS}
        self._lm = lm
        self._reserve = reserve

    @staticmethod
    def _check(kwargs: dict[str, Any]) -> None:
        rejected = sorted(kwargs.keys() - _LM_CALL_INPUTS - SANDBOX_LM_KWARGS)
        if rejected:
            raise TypeError(
                f"Sandboxed code may not set LM option(s) {rejected}; allowed per call: {sorted(SANDBOX_LM_KWARGS)}."
            )

    def _admit(self, kwargs: dict[str, Any]) -> None:
        self._check(kwargs)
        if self._reserve is not None:
            self._reserve(1)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._admit(kwargs)
        return self._lm(*args, **kwargs)

    async def acall(self, *args: Any, **kwargs: Any) -> Any:
        self._admit(kwargs)
        return await self._lm.acall(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._lm.forward(*args, **kwargs)

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
    """

    def __init__(
        self,
        tools: dict[str, Any],
        interpreter_factory: Callable[[], Any] | None,
        max_predictor_calls: int | None,
        *,
        custom_types: dict[str, type] | None = None,
        lm: Any = None,
        originals: dict[str, Any] | None = None,
    ) -> None:
        self._tools = tools
        self._interpreter_factory = interpreter_factory
        self._max_predictor_calls = max_predictor_calls
        self._custom_types = custom_types
        self._lm = lm
        self._originals = originals or {}
        self._predictors: dict[str, Any] = {}
        self._calls = 0
        self._lm_error: tuple[LMError, str] | None = None

    def install(self, interpreter: Any) -> None:
        """Register the tools the shim calls, then install the ``dspy`` facade in the sandbox."""
        interpreter.tools.update({CONSTRUCT_TOOL: self.construct, CALL_TOOL: self.call})
        interpreter.execute(SHIM_SETUP)

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
            raise CodeInterpreterError(f"Sandboxed forward exceeded its predictor-call budget ({budget}).")
        predictor = self._predictors.get(handle)
        if predictor is None:
            raise CodeInterpreterError(f"Unknown predictor handle: {handle!r}")
        restored = {k: _restore_custom_types(v, self._originals) for k, v in (inputs or {}).items()}
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
        return dspy.context(lm=lm if isinstance(lm, SandboxLM) else SandboxLM(lm))

    def _build_predictor(self, kind: str, signature: Any, kwargs: dict[str, Any] | None) -> Any:
        cls = getattr(dspy, kind)
        extra = {k: self._decode_tools(v) for k, v in (kwargs or {}).items()}
        # A code-executing sub-predictor runs its inner code on the host module's backend; sandbox
        # code can't set this itself, since a live interpreter can't cross the boundary.
        if extra.get("interpreter_factory", FACTORY_MARKER) == FACTORY_MARKER:
            extra.pop("interpreter_factory", None)
            if self._interpreter_factory and _accepts_interpreter_factory(cls):
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
