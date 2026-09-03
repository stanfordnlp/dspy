"""Host side of the dspy.Flex sandbox bridge.

A ``Flex`` never executes its optimizer-authored ``module_src`` in the host process. Instead,
every ``forward``:

1. creates a fresh interpreter from the Flex's ``interpreter_factory`` (``BridgeRuntime.forward``);
2. installs the sandbox dspy facade (``dspy.primitives.facade``) over the Flex's tools, interpreter
   factory, predictor-call budget, and custom types;
3. executes ``module_src`` and drives its ``forward`` with the call's inputs.

User tools passed to ``dspy.Flex(tools=...)`` are registered by name so sandbox code can call them
directly; the callables themselves stay on the host. The final ``dspy.Prediction`` is parsed against
the Flex signature's declared output types on the way out (``BridgeRuntime._to_prediction``).
"""

from __future__ import annotations

import ast
import json
import logging
from typing import Any, Callable

from pydantic import TypeAdapter

import dspy
from dspy.adapters.utils import annotation_allows_none, parse_value
from dspy.primitives.code_interpreter import CodeInterpreterError, _create_interpreter
from dspy.primitives.facade import (
    FacadeInvocation,
    _collect_custom_type_originals,
    _restoring_entrypoint,
    _tool_entrypoint,
)

logger = logging.getLogger(__name__)

# Variable/identifier names used in the per-forward driver code (namespaced to avoid clashing with
# whatever the optimizer-authored module uses).
_INPUTS_VAR = "__dspy_flex_inputs"
_INSTANCE_VAR = "__dspy_flex_instance"
_OUT_VAR = "__dspy_flex_out"
_JSON_VAR = "__dspy_flex_json"


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


class BridgeRuntime:
    """Host side of the dspy.Flex bridge: runs the bound ``module_src`` for one ``Flex``.

    Each ``forward`` creates a fresh interpreter from the factory, executes the shim and the
    bound source, runs the module instance, and shuts the interpreter down. Predictors are built
    host-side but held by that forward's ``FacadeInvocation`` — never set on the ``Flex`` — so nothing
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

    def invocation(self, originals: dict[str, Any] | None = None) -> FacadeInvocation:
        """A fresh per-forward invocation over tools, factory, budget, and custom types."""
        ctx = self._flex._flex_ctx
        return FacadeInvocation(
            ctx.context_names(),
            self._factory,
            self._max_predictor_calls,
            custom_types=ctx.custom_types(),
            originals=originals,
        )

    def forward(self, inputs: dict[str, Any]) -> Any:
        originals: dict[str, Any] = {}
        for v in inputs.values():
            _collect_custom_type_originals(v, originals)

        invocation = self.invocation(originals)
        interp = _create_interpreter(self._factory)
        try:
            # User tools callable by name in the sandbox.
            interp.tools.update(
                {name: _restoring_entrypoint(fn, originals) for name, fn in self._tool_callables().items()}
            )
            invocation.install(interp)
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
