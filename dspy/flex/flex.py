from __future__ import annotations

import copy
from typing import Any, Callable

from dspy.flex.ctx import FlexContext
from dspy.primitives.code_interpreter import CodeInterpreter, _validate_interpreter_factory
from dspy.primitives.module import Module
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.utils.annotation import experimental


@experimental(version="3.3.0b2")
class Flex(Module):
    """A module whose implementation is optimizable code, not just a prompt.

    Construct it like any module (``dspy.Flex(MySignature)``). It starts as a baseline that delegates
    to a single ``dspy.Predict`` over the signature — or ``dspy.RLM`` when ``tools`` are given, so the
    baseline can call them — and is marked ``_code_optimizable``, so ``dspy.GEPA`` can rewrite its
    source — a single ``dspy.Module`` subclass, exposed as ``module_src`` — into decomposed predictors
    plus plain Python instead of only tuning instructions.

    The optimizer-authored code runs inside an interpreter. ``Flex`` never
    runs it in the host Python process. ``interpreter_factory`` defaults to ``dspy.PythonInterpreter``
    (Deno/Pyodide) and must be a zero-argument callable returning a new ``CodeInterpreter``.
    The optimizer-authored glue runs isolated; only provided-tool calls, predictor construction,
    and predictor calls bridge back to the host, which makes the real LM calls.

    Args:
        signature: A ``dspy.Signature`` class or string declaring inputs/outputs.
        tools: ``dspy.Tool`` instances or named callables.
        interpreter_factory: Zero-argument callable returning a ``CodeInterpreter`` for each forward
            pass. Defaults to ``dspy.PythonInterpreter`` (sandbox, requires Deno).
        max_predictor_calls: Cap on bridged LM calls per ``forward``; ``None`` disables it.
    """

    # dspy.GEPA reads this marker (duck-typed) to know it may rewrite the module's code.
    _code_optimizable: bool = True

    def __init__(
        self,
        signature: Any,
        *,
        tools: list[Any] | None = None,
        interpreter_factory: Callable[[], CodeInterpreter] = PythonInterpreter,
        max_predictor_calls: int | None = 100,
    ):
        super().__init__()

        from dspy.signatures.signature import ensure_signature

        self._signature_cls = ensure_signature(signature)
        self._name = getattr(self._signature_cls, "__name__", None) or "Flex"
        self._flex_ctx = FlexContext(signature_cls=self._signature_cls, tools=list(tools or []))

        self._module_src: str | None = None
        self._attached_names: list[str] = []

        self._bridge: Any = None
        _validate_interpreter_factory(interpreter_factory)
        self._interpreter_factory = interpreter_factory
        self._max_predictor_calls = max_predictor_calls

        from dspy.flex.bridge import BridgeRuntime

        self._bridge = BridgeRuntime(self, self._interpreter_factory, self._max_predictor_calls)
        self._bind_code(self._baseline_src())

    @property
    def signature(self) -> Any:
        return self._signature_cls

    @property
    def module_src(self) -> str | None:
        return self._module_src

    def dump_state(self, json_mode: bool = True) -> dict[str, Any]:
        state = super().dump_state(json_mode=json_mode)
        state["module_src"] = self._module_src
        return state

    def load_state(self, state: dict[str, Any], *, allow_unsafe_lm_state: bool = False) -> None:
        state = dict(state) if isinstance(state, dict) else state
        module_src = state.pop("module_src", None) if isinstance(state, dict) else None
        if module_src:
            self._bind_code(module_src)
        if state:
            super().load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)

    def _baseline_src(self) -> str:
        """Starting source: one predictor over the whole signature, wrapped in a ``dspy.Module``.

        Uses ``dspy.RLM`` when tools are provided; otherwise a single ``dspy.Predict``.
        """
        cls: Any = self._signature_cls
        sig_str = self._flex_ctx.render_signature_string()
        returns = ", ".join(f"{name}=result.{name}" for name in cls.output_fields)
        instructions = (getattr(cls, "instructions", "") or "").strip()
        sig_arg = f"dspy.Signature({sig_str!r}, {instructions!r})" if instructions else repr(sig_str)
        tool_names = list(self._flex_ctx.context_names())
        if tool_names:
            attr, ctor = "rlm", f"dspy.RLM({sig_arg}, tools=[{', '.join(tool_names)}])"
        else:
            attr, ctor = "predict", f"dspy.Predict({sig_arg})"
        return (
            f"class {self._class_name()}(dspy.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.{attr} = {ctor}\n"
            "\n"
            "    def forward(self, **inputs):\n"
            f"        result = self.{attr}(**inputs)\n"
            f"        return dspy.Prediction({returns})"
        )

    def _class_name(self) -> str:
        base = self._name if (self._name and self._name.isidentifier()) else "Flex"
        return f"{base}Module"

    def _bind_code(self, module_src: str) -> None:
        """Bind ``module_src``: The sandbox constructs the predictors and bridges them back to attach onto ``self``.
        ``forward`` runs in the sandbox."""
        for old_name in self._attached_names:
            if hasattr(self, old_name):
                delattr(self, old_name)
        self._attached_names = []

        self._bridge.bind(module_src)
        self._module_src = module_src
        self._bridge.ensure_initialized()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the bound ``forward`` inside the interpreter."""
        if args:
            raise TypeError("dspy.Flex accepts keyword inputs only")
        return self._bridge.forward(kwargs)

    def close(self) -> None:
        """Shut down any interpreter sessions this Flex created."""
        bridge = getattr(self, "_bridge", None)
        if bridge is not None:
            bridge.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __deepcopy__(self, memo):
        # Each copy needs its own interpreter sessions. Sharing `_bridge` by reference would let a
        # throwaway copy's __del__ tear down the original's live session, so copy everything else and
        # give the copy a fresh session-less bridge that reuses the already-copied predictors.
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == "_bridge":
                continue
            try:
                setattr(new, key, copy.deepcopy(value, memo))
            except Exception:
                setattr(new, key, value)
        new._bridge = None
        if new._module_src is not None:
            from dspy.flex.bridge import BridgeRuntime

            bridge = BridgeRuntime(new, new._interpreter_factory, new._max_predictor_calls)
            bridge.bind(new._module_src)
            if self._bridge is not None:
                bridge._registry = dict(self._bridge._registry)
            new._bridge = bridge
        return new
