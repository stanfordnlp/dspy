"""Host side of the dspy.Flex sandbox bridge.

A ``Flex`` never executes its optimizer-authored ``module_src`` in the host process. Instead,
every ``forward``:

1. creates a fresh interpreter from the Flex's ``interpreter_factory`` (``BridgeRuntime.forward``);
2. injects the sandbox-side shim (``_sandbox_shim.py``), which fakes a tiny ``dspy`` module whose
   predictor constructors and calls are proxies;
3. executes ``module_src`` and drives its ``forward`` with the call's inputs.

Only three things cross the sandbox boundary, all as JSON over the interpreter's tool-call
protocol (``CodeInterpreter.tools``):

- ``__dspy_construct__``: the shim asks the host to build a real predictor (``_Invocation.construct``).
  The host constructs it from the serialized signature/kwargs and returns a string handle.
- ``__dspy_call__``: the shim runs a predictor by handle (``_Invocation.call``); the host makes the
  real LM call and returns the prediction's fields as JSON.
- user tools passed to ``dspy.Flex(tools=...)``, registered by name so sandbox code can call them
  directly; the callables themselves stay on the host.

All per-forward state, including the constructed predictors, the predictor-call budget, custom-type
originals, and the last LM infrastructure error, lives in a ``_Invocation`` created for that
forward, never on the shared ``Flex``, so concurrent forwards (e.g. threaded evaluation) are
isolated. The final ``dspy.Prediction`` is parsed against the Flex signature's declared output
types on the way out (``BridgeRuntime._to_prediction``).
"""

from __future__ import annotations

import ast
import functools
import inspect
import json
import logging
import types
from pathlib import Path
from typing import Any, Callable, Union, get_args, get_origin

from pydantic import TypeAdapter
from pydantic_core import PydanticSerializationError, to_jsonable_python

import dspy
from dspy import CodeInterpreterError
from dspy.adapters.types.base_type import Type as _CustomType
from dspy.adapters.utils import parse_value
from dspy.primitives.code_interpreter import _create_interpreter
from dspy.signatures.signature import make_signature
from dspy.utils.exceptions import LMError

logger = logging.getLogger(__name__)

# Tool names the shim uses to call the host; must match the names registered in BridgeRuntime.
CONSTRUCT_TOOL = "__dspy_construct__"
CALL_TOOL = "__dspy_call__"

# Predictors the sandbox shim may construct and the host builds
BRIDGEABLE_KINDS = ("Predict", "ChainOfThought", "RLM", "CodeAct", "ProgramOfThought", "ReAct", "ReActV2")
# The shim's dspy.Signature(...) emits this marker so the host can rebuild a Signature.
SIGNATURE_MARKER = "__dspy_sig__"
# The shim passes tools by name (callables can't cross the JSON boundary); the host resolves the name
# back to the real tool object passed to dspy.Flex(tools=...).
TOOL_MARKER = "__dspy_tool__"
# Variable/identifier names used in the per-forward driver code (namespaced to avoid clashing with
# whatever the optimizer-authored module uses).
_INPUTS_VAR = "__dspy_flex_inputs"
_INSTANCE_VAR = "__dspy_flex_instance"
_OUT_VAR = "__dspy_flex_out"
_JSON_VAR = "__dspy_flex_json"


# The sandbox-side dspy shim, injected as text into each per-forward interpreter.
SHIM_SETUP = (Path(__file__).parent / "_sandbox_shim.py").read_text(encoding="utf-8")


def parse_module_class_name(module_src: str) -> str:
    """Return the name of the generated ``dspy.Module`` subclass in ``module_src``."""
    tree = ast.parse(module_src)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    with_forward = [
        c for c in classes if any(isinstance(b, ast.FunctionDef) and b.name == "forward" for b in c.body)
    ]
    chosen = with_forward or classes
    if not chosen:
        raise CodeInterpreterError("module_src must define a dspy.Module subclass with a `forward` method")
    return chosen[0].name


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


def _is_field_optional(annotation: Any) -> bool:
    """True if the field annotation is optional, a union admitting None."""
    return annotation is type(None) or (
        get_origin(annotation) in (Union, types.UnionType) and type(None) in get_args(annotation)
    )


class _Invocation:
    """Per-forward bridge state: the predictors this forward constructed (keyed by the sandbox
    attribute name), its predictor-call budget, and the original custom-type objects to restore
    on bridged calls. Each forward gets its own ``_Invocation``."""

    def __init__(self, runtime: BridgeRuntime, originals: dict[str, Any]) -> None:
        self._runtime = runtime
        self._originals = originals
        self._predictors: dict[str, Any] = {}
        self._calls = 0
        self._lm_error: tuple[LMError, str] | None = None

    def construct(self, kind: str, signature: Any, attr_name: str, kwargs: dict[str, Any] | None = None) -> str:
        self._lm_error = None
        if kind not in BRIDGEABLE_KINDS:
            raise CodeInterpreterError(
                f"dspy.{kind} is not supported inside a sandboxed dspy.Flex yet "
                f"(bridgeable: {', '.join(BRIDGEABLE_KINDS)})"
            )
        self._predictors[attr_name] = self._runtime._build_predictor(kind, signature, kwargs)
        return attr_name

    def call(self, handle: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        self._lm_error = None
        self._calls += 1
        budget = self._runtime._max_predictor_calls
        if budget is not None and self._calls > budget:
            raise CodeInterpreterError(
                f"Sandboxed dspy.Flex forward exceeded its predictor-call budget "
                f"({budget}). Raise max_predictor_calls if this is expected."
            )
        predictor = self._predictors.get(handle)
        if predictor is None:
            raise CodeInterpreterError(f"Unknown predictor handle: {handle!r}")
        restored = {k: _restore_custom_types(v, self._originals) for k, v in (inputs or {}).items()}
        try:
            return prediction_to_fields(predictor(**restored))
        except LMError as e:
            tag = f"[dspy bridge lm-error #{self._calls}]"
            self._lm_error = (e, tag)
            raise CodeInterpreterError(f"{tag} {type(e).__name__}: {e}") from e


class BridgeRuntime:
    """Host side of the dspy.Flex bridge: runs the bound ``module_src`` for one ``Flex``.

    Each ``forward`` creates a fresh interpreter from the factory, executes the shim and the
    bound source, runs the module instance, and shuts the interpreter down. Predictors are built
    host-side but held by that forward's ``_Invocation`` — never set on the ``Flex`` — so nothing
    outlives the call and parallel forwards stay isolated. ``_max_predictor_calls`` caps bridged
    predictor calls per forward.

    User ``tools`` are registered with the interpreter, callable by name from the sandbox, and
    resolved by name when handed to a bridged sub-predictor; functions defined inside the
    generated module stay in the sandbox. A code-executing sub-predictor (RLM, CodeAct,
    ProgramOfThought) receives the same interpreter factory, so its inner code runs on the
    backend chosen for the Flex.
    """

    def __init__(self, flex: Any, factory: Callable[[], Any], max_predictor_calls: int | None = 100) -> None:
        self._flex = flex
        self._factory = factory
        self._max_predictor_calls = max_predictor_calls
        self._module_src: str | None = None
        self._class_name: str | None = None

    def bind(self, module_src: str) -> None:
        """Record new source; it runs at the next ``forward``."""
        self._module_src = module_src
        self._class_name = parse_module_class_name(module_src)

    def forward(self, inputs: dict[str, Any]) -> Any:
        originals: dict[str, Any] = {}
        for v in inputs.values():
            _collect_custom_type_originals(v, originals)

        invocation = _Invocation(self, originals)
        interp = _create_interpreter(self._factory)
        try:
            interp.tools.update({CONSTRUCT_TOOL: invocation.construct, CALL_TOOL: invocation.call})
            # User tools callable by name in the sandbox.
            interp.tools.update(
                {name: _restoring_entrypoint(fn, originals) for name, fn in self._tool_callables().items()}
            )
            interp.execute(SHIM_SETUP)
            interp.execute(self._module_src)  # defines the class in the sandbox
            code = (
                f"{_INSTANCE_VAR} = {self._class_name}()\n"
                f"{_OUT_VAR} = {_INSTANCE_VAR}.forward(**{_INPUTS_VAR})\n"
                f"import json as {_JSON_VAR}\n"
                f"{_JSON_VAR}.dumps({_OUT_VAR}._fields if hasattr({_OUT_VAR}, '_fields') else {_OUT_VAR})"
            )
            result = interp.execute(code, variables={_INPUTS_VAR: dict(inputs)})
        except CodeInterpreterError as e:
            if invocation._lm_error is not None:
                lm_error, tag = invocation._lm_error
                if tag in str(e):
                    raise lm_error from e
            raise
        finally:
            try:
                interp.shutdown()
            except Exception:
                logger.warning("dspy.Flex: interpreter.shutdown() raised after forward", exc_info=True)
        if not isinstance(result, str) or not result:
            raise CodeInterpreterError(
                "Sandboxed forward returned no serializable result; the generated forward must return "
                f"a dspy.Prediction (got {result!r})"
            )
        return self._to_prediction(json.loads(result))

    def _to_prediction(self, fields: dict[str, Any]) -> Any:
        signature = self._flex.signature
        out = dict(fields)
        filled: set[str] = set()
        missing: list[str] = []
        for name, field in signature.output_fields.items():
            if name in out:
                continue
            if not field.is_required():
                out[name] = field.get_default(call_default_factory=True)
                filled.add(name)
            elif _is_field_optional(field.annotation):
                out[name] = None
                filled.add(name)
            else:
                missing.append(name)
        if missing:
            raise CodeInterpreterError(
                f"Sandboxed forward returned a dspy.Prediction missing required output field(s) "
                f"{missing}; the signature declares {list(signature.output_fields)}."
            )
        for name, field in signature.output_fields.items():
            if name in filled:
                continue
            try:
                if out[name] is None:
                    out[name] = TypeAdapter(field.annotation).validate_python(None)
                else:
                    out[name] = parse_value(out[name], field.annotation)
            except Exception as e:
                raise CodeInterpreterError(
                    f"Sandboxed forward returned {out[name]!r} for output field {name!r}, which is "
                    f"not a valid {field.annotation}: {e}"
                ) from e
        return dspy.Prediction(**out)

    # -- host-side bridge callbacks --------------

    def _tool_callables(self) -> dict[str, Callable[..., Any]]:
        """User tools as ``name -> callable`` to register so sandbox code can call them."""
        return {name: _tool_entrypoint(tool) for name, tool in self._flex._flex_ctx.context_names().items()}

    def _resolve_tool(self, name: str) -> Any:
        ctx = self._flex._flex_ctx.context_names()
        if name not in ctx:
            raise CodeInterpreterError(
                f"Sandboxed code referenced tool {name!r}, which was not passed to dspy.Flex(tools=...). "
                "Tools authored inside the generated module cannot be handed to a bridged sub-predictor."
            )
        return ctx[name]

    def _decode_tools(self, value: Any) -> Any:
        """Turn the shim's tool name-markers back into the real host tool objects."""
        if isinstance(value, dict) and TOOL_MARKER in value:
            return self._resolve_tool(value[TOOL_MARKER])
        if isinstance(value, list):
            return [self._decode_tools(v) for v in value]
        if isinstance(value, dict):
            return {k: self._decode_tools(v) for k, v in value.items()}
        return value

    def _sub_interpreter_factory(self) -> Any:
        return self._factory

    def _build_predictor(self, kind: str, signature: Any, kwargs: dict[str, Any] | None) -> Any:
        cls = getattr(dspy, kind)
        extra = {k: self._decode_tools(v) for k, v in (kwargs or {}).items()}
        # A code-executing sub-predictor should run its inner code in the backend chosen for Flex, so
        # hand it the Flex interpreter factory (it makes and tears down a fresh interpreter per forward).
        # The sandbox code can't set this itself, since a live interpreter can't cross the boundary.
        if "interpreter_factory" not in extra and _accepts_interpreter_factory(cls):
            factory = self._sub_interpreter_factory()
            if factory is not None:
                extra["interpreter_factory"] = factory
        return cls(_resolve_signature(signature, self._flex._flex_ctx.custom_types()), **extra)
