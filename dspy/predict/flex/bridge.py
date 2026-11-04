"""Host side of the dspy.Flex sandbox bridge.

A ``Flex`` never executes its optimizer-authored ``module_src`` in the host process. Instead,
every ``forward``:

1. creates a fresh interpreter from the Flex's ``interpreter_factory``, or from
   ``dspy.settings.interpreter_factory`` when Flex uses its default (``BridgeRuntime.forward``);
2. imports the real dspy there, wired to the host by a ``SandboxSession`` (``dspy.primitives.facade``) over the
   Flex's tools, custom types and LM-call budget;
3. executes ``module_src`` and drives its ``forward`` with the call's inputs.

User tools passed to ``dspy.Flex(tools=...)`` are registered by name so sandbox code can call them
directly; the callables themselves stay on the host. The final ``dspy.Prediction`` is parsed against
the Flex signature's declared output types on the way out (``BridgeRuntime._to_prediction``).
"""

from __future__ import annotations

import ast
import contextlib
import functools
import json
import logging
from typing import Any, Callable

from pydantic import TypeAdapter

import dspy
from dspy.adapters.types.base_type import Type as _CustomType
from dspy.adapters.utils import annotation_allows_none
from dspy.predict.flex.ctx import render_annotation
from dspy.primitives.code_interpreter import CodeInterpreterError, _create_interpreter
from dspy.primitives.facade import SandboxSession, from_wire
from dspy.primitives.sandbox_guest import encode_value

logger = logging.getLogger(__name__)

# The per-forward inputs in the sandbox (namespaced to avoid clashing with whatever the module uses).
_INPUTS_VAR = "__dspy_flex_inputs"


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


def _tool_entrypoint(tool: Any) -> Callable[..., Any]:
    """A callable for the interpreter's tool registry."""
    func = getattr(tool, "func", None)
    if func is None:
        return tool

    @functools.wraps(func)
    def entrypoint(**kwargs: Any) -> Any:
        return tool(**kwargs)

    return entrypoint


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _restore(value: Any, originals: dict[str, Any]) -> Any:
    """Swap custom-type values that came back from the sandbox for the host objects they were made from."""
    if isinstance(value, dict):
        original = originals.get(_key(value))
        return original if original is not None else {k: _restore(v, originals) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore(v, originals) for v in value]
    return value


def _restoring_entrypoint(fn: Callable[..., Any], originals: dict[str, Any]) -> Callable[..., Any]:
    """Wrap a tool entrypoint so custom-type inputs arrive as the original objects."""

    @functools.wraps(fn)
    def entrypoint(**kwargs: Any) -> Any:
        return fn(**{k: _restore(v, originals) for k, v in kwargs.items()})

    return entrypoint


def _collect_custom_type_originals(value: Any, out: dict[str, Any]) -> None:
    """Record custom-type instances by the form they cross the boundary in, recursing into containers."""
    if isinstance(value, _CustomType):
        out[_key(encode_value(value))] = value
    elif isinstance(value, dict):
        for v in value.values():
            _collect_custom_type_originals(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _collect_custom_type_originals(v, out)


class _Budget:
    """The LM calls one forward may make, counted by the host as the sandbox requests them."""

    def __init__(self, limit: int | None) -> None:
        self._limit = limit
        self._count = 0

    def reserve(self, n: int = 1) -> None:
        self._count += n
        if self._limit is not None and self._count > self._limit:
            raise CodeInterpreterError(
                f"Sandboxed forward exceeded its LM-call budget ({self._limit}). "
                "Raise max_predictor_calls if this is expected."
            )


class BridgeRuntime:
    """Host side of the dspy.Flex bridge: runs the bound ``module_src`` for one ``Flex``.

    Each ``forward`` creates a fresh interpreter from the factory, imports dspy there, executes the bound
    source, runs the module instance, and shuts the interpreter down, so nothing outlives the call and
    parallel forwards stay isolated. Every LM call the sandbox makes runs on the host and counts against
    ``_max_predictor_calls``.

    User ``tools`` are registered with the interpreter and callable by name from the sandbox, where they
    are ``dspy.Tool``s that predictors built there can use. Each code-executing module built in the sandbox,
    like an RLM, gets a fresh interpreter from the same factory, shut down with the forward.
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

    def session(self) -> SandboxSession:
        """A fresh per-forward session over tools, custom types, and the LM-call budget."""
        ctx = self._flex._flex_ctx
        custom = ctx.custom_types()
        inputs = {}
        for name, field in self._flex.signature.input_fields.items():
            with contextlib.suppress(ValueError):
                inputs[name] = render_annotation(field.annotation, custom)
        return SandboxSession(
            ctx.context_names(),
            custom_types=custom,
            inputs=inputs,
            reserve=_Budget(self._max_predictor_calls).reserve,
            interpreter_factory=self._factory,
        )

    def forward(self, inputs: dict[str, Any]) -> Any:
        originals: dict[str, Any] = {}
        for v in inputs.values():
            _collect_custom_type_originals(v, originals)

        session = self.session()
        interp = _create_interpreter(self._factory)
        try:
            # User tools callable by name in the sandbox.
            interp.tools.update(
                {name: _restoring_entrypoint(fn, originals) for name, fn in self._tool_callables().items()}
            )
            session.install(interp, import_dspy=True)
            interp.execute(self._module_src)  # defines the class in the sandbox
            code = (
                "from dspy.primitives.sandbox_guest import run_forward as __dspy_run_forward\n"
                f"__dspy_run_forward(globals(), {self._class_name!r}, {_INPUTS_VAR})"
            )
            with session.scope():
                result = interp.execute(code, variables={_INPUTS_VAR: encode_value(dict(inputs))})
        except CodeInterpreterError as e:
            if session.lm_error is not None:
                lm_error, tag = session.lm_error
                if tag in str(e):
                    raise lm_error from e
            raise
        finally:
            session.close()
            try:
                interp.shutdown()
            except Exception:
                logger.warning("dspy.Flex: interpreter.shutdown() raised after forward", exc_info=True)
        if not isinstance(result, str) or not result:
            raise CodeInterpreterError(
                "Sandboxed forward returned no serializable result; the generated forward must return "
                f"a dspy.Prediction (got {result!r})"
            )
        return self._to_prediction(json.loads(result)["fields"])

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
            elif annotation_allows_none(field.annotation):
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
                    out[name] = from_wire(out[name], field.annotation)
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
