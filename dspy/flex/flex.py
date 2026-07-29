from __future__ import annotations

from typing import Any, Callable

from dspy.dsp.utils.settings import settings
from dspy.flex.ctx import FlexContext
from dspy.predict.parameter import Parameter
from dspy.primitives.code_interpreter import CodeInterpreter, _validate_interpreter_factory
from dspy.primitives.module import Module
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.utils.annotation import experimental


@experimental(version="3.3.0b2")
class Flex(Module, Parameter):
    """A module whose implementation is optimizable code, not just a prompt.

    Construct it like any module (``dspy.Flex(MySignature)``). It starts as a baseline that delegates
    to a single ``dspy.Predict`` over the signature — or ``dspy.RLM`` when ``tools`` are given, so the
    baseline can call them. ``dspy.GEPA`` recognizes ``Flex`` instances by type and rewrites their
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
        self.lm = None

        _validate_interpreter_factory(interpreter_factory)
        self._interpreter_factory = interpreter_factory
        self._max_predictor_calls = max_predictor_calls

        from dspy.flex.bridge import BridgeRuntime

        self._bridge: BridgeRuntime = BridgeRuntime(self, self._interpreter_factory, self._max_predictor_calls)
        self._bind_code(self._baseline_src())

    @property
    def signature(self) -> Any:
        return self._signature_cls

    @property
    def module_src(self) -> str | None:
        return self._module_src

    def dump_state(self, json_mode: bool = True) -> dict[str, Any]:
        return {
            "module_src": self._module_src,
            "lm": self.lm.dump_state() if self.lm else None,
        }

    def load_state(self, state: dict[str, Any], *, allow_unsafe_lm_state: bool = False) -> None:
        module_src = state.get("module_src")
        if module_src:
            self._bind_code(module_src)
        lm_state = state.get("lm")
        if lm_state:
            from dspy.clients.base_lm import BaseLM
            from dspy.predict.predict import _sanitize_lm_state

            sanitized = _sanitize_lm_state(lm_state, allow_unsafe_lm_state)
            self.lm = (
                BaseLM.load_state(sanitized, allow_custom_lm_class=allow_unsafe_lm_state) if sanitized else None
            )
        else:
            self.lm = None

    def reset(self) -> None:
        """Clear the LM; ``module_src`` is kept, and predictors are rebuilt from it each forward."""
        self.lm = None

    def named_predictors(self):
        """A Flex's update unit is only its ``module_src``."""
        return []

    def set_lm(self, lm) -> None:
        self.lm = lm

    def get_lm(self):
        return self.lm

    def __repr__(self):
        return f"{self.__class__.__name__}({self._name})"

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
        self._bridge.bind(module_src)
        self._module_src = module_src

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the bound ``forward`` inside the interpreter."""
        if args:
            raise TypeError("dspy.Flex accepts keyword inputs only")
        if self.lm is not None:
            with settings.context(lm=self.lm):
                return self._bridge.forward(kwargs)
        return self._bridge.forward(kwargs)
