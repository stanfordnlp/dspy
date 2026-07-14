from __future__ import annotations

import copy
import logging
import types
from typing import Any, Callable

import dspy
from dspy.flex.ctx import FlexContext
from dspy.primitives.code_interpreter import CodeInterpreter
from dspy.primitives.module import Module
from dspy.utils.annotation import experimental

logger = logging.getLogger(__name__)


def _exec_source(source: str, context_names: dict[str, Any] | None = None) -> dict[str, Any]:
    code = compile(source, filename="<flex>", mode="exec")
    globals_ns: dict[str, Any] = dict(context_names or {})
    exec(code, globals_ns)
    return globals_ns


@experimental(version="3.3.0b2")
class Flex(Module):
    """A module whose implementation is optimizable code, not just a prompt.

    Construct it like any module (``dspy.Flex(MySignature)``). It starts as a baseline that delegates
    to a single ``dspy.Predict`` over the signature — or ``dspy.RLM`` when ``tools`` are given, so the
    baseline can call them — and is marked ``_code_optimizable``, so ``dspy.GEPA`` can rewrite its
    source — a single ``dspy.Module`` subclass, exposed as ``module_src`` — into decomposed predictors
    plus plain Python instead of only tuning instructions.

    Pass ``interpreter`` to run the generated code inside a sandbox rather than with in-process
    ``exec``. The optimizer-authored glue (control flow, string work, imports) runs isolated; only
    predictor construction and predictor calls bridge back to the host, which makes the real LM
    calls. Give a zero-arg factory (``interpreter=lambda: dspy.PythonInterpreter()``) so parallel
    evaluations each get their own session; a bare instance is shared, and Flex warns about it. The
    interpreter is a runtime dependency like the LM: it is not serialized, so re-supply it before
    ``load``. Any ``CodeInterpreter`` backend works — see ``dspy/flex/bridge.py``.

    Args:
        signature: A ``dspy.Signature`` class or string declaring inputs/outputs.
        tools: ``dspy.Tool`` instances or named callables.
        interpreter: A ``CodeInterpreter`` instance or zero-arg factory. When set, generated code
            runs in the sandbox.
        max_predictor_calls: Cap on bridged LM calls per sandboxed ``forward`` (a runaway guard);
            ``None`` disables it. Ignored without an ``interpreter``.
    """

    # dspy.GEPA reads this marker (duck-typed) to know it may rewrite the module's code.
    _code_optimizable: bool = True

    def __init__(
        self,
        signature: Any,
        *,
        tools: list[Any] | None = None,
        interpreter: CodeInterpreter | Callable[[], CodeInterpreter] | None = None,
        max_predictor_calls: int | None = 100,
    ):
        super().__init__()

        from dspy.signatures.signature import ensure_signature

        self._signature_cls = ensure_signature(signature)
        self._name = getattr(self._signature_cls, "__name__", None) or "Flex"
        self._flex_ctx = FlexContext(signature_cls=self._signature_cls, tools=list(tools or []))

        self._module_src: str | None = None
        self._attached_names: list[str] = []
        self._forward_impl: Any = None

        self._interpreter_factory = self._normalize_interpreter(interpreter)
        # A bare instance is shared across forward() calls, so it can't be handed to a sub-predictor
        # that shuts its interpreter down after forward (see BridgeRuntime._sub_interpreter_factory).
        self._interpreter_shared = isinstance(interpreter, CodeInterpreter)
        self._max_predictor_calls = max_predictor_calls
        self._bridge: Any = None
        if self._interpreter_factory is not None:
            from dspy.flex.bridge import BridgeRuntime

            self._bridge = BridgeRuntime(self, self._interpreter_factory, self._max_predictor_calls)

        self._bind_code(self._baseline_src())

    @staticmethod
    def _normalize_interpreter(
        interpreter: CodeInterpreter | Callable[[], CodeInterpreter] | None,
    ) -> Callable[[], CodeInterpreter] | None:
        """Normalize an interpreter instance-or-factory to a zero-arg factory (or ``None``)."""
        if interpreter is None:
            return None
        # Check for an instance before a callable: an interpreter may itself be callable, but it's
        # an instance to wrap, not a factory to use directly.
        if isinstance(interpreter, CodeInterpreter):
            logger.warning(
                "dspy.Flex received a CodeInterpreter instance; it will be shared across all forward() "
                "calls. PythonInterpreter is not thread-safe and is stateful, so this is unsafe under "
                "parallel evaluation/optimization. Pass a zero-arg factory "
                "(interpreter=lambda: dspy.PythonInterpreter(...)) for isolation."
            )
            return lambda: interpreter
        if callable(interpreter):
            return interpreter
        raise TypeError(
            "interpreter must be a CodeInterpreter instance or a zero-arg callable returning one, "
            f"got {type(interpreter).__name__}"
        )

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

        Uses ``dspy.RLM`` when tools are provided, so the baseline can call them through its REPL;
        otherwise a single ``dspy.Predict``, which runs with just an LM and no code interpreter.
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
        """Bind ``module_src``: in-process ``exec`` by default, or via the sandbox bridge when an
        ``interpreter`` was provided. Either way the resulting predictors are attached onto ``self``."""
        for old_name in self._attached_names:
            if hasattr(self, old_name):
                delattr(self, old_name)
        self._attached_names = []

        if self._bridge is not None:
            self._bind_code_bridged(module_src)
            return

        ctx_names = self._flex_ctx.context_names()
        ctx_names["dspy"] = dspy

        ns = _exec_source(module_src, context_names=ctx_names)
        impl_cls = _find_module_class(ns)
        forward_fn = impl_cls.__dict__.get("forward")
        if not callable(forward_fn):
            raise RuntimeError("module_src's dspy.Module subclass must define a `forward` method")

        impl = impl_cls()  # runs __init__, constructing predictors only
        baseline_keys = _bare_module_keys()
        for key, value in list(impl.__dict__.items()):
            if key in baseline_keys:
                continue  # internal dspy.Module bookkeeping, not user-defined state
            setattr(self, key, value)
            self._attached_names.append(key)

        self._forward_impl = types.MethodType(forward_fn, self)
        self._module_src = module_src

    def _bind_code_bridged(self, module_src: str) -> None:
        """Bind ``module_src`` for sandboxed execution: the host never ``exec``s the source. The
        sandbox constructs predictors and bridges them back to attach onto ``self``, and ``forward``
        runs in the sandbox. We build eagerly so ``named_predictors()``/``load_state`` see them.
        """
        self._forward_impl = None  # the bridge runs forward, not an in-process method
        self._bridge.bind(module_src)
        self._module_src = module_src
        self._bridge.ensure_initialized()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the bound ``forward`` (in-process, or inside the sandbox when an interpreter is set)."""
        if self._bridge is not None:
            if args:
                raise TypeError("dspy.Flex with an interpreter accepts keyword inputs only")
            return self._bridge.forward(kwargs)
        if self._forward_impl is None:
            raise RuntimeError(f"dspy.Flex {self._name!r}: no implementation bound.")
        return self._forward_impl(*args, **kwargs)

    def close(self) -> None:
        """Shut down any sandbox interpreter sessions this Flex created. Safe to call repeatedly."""
        if self._bridge is not None:
            self._bridge.shutdown()

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
        # Each copy needs its own sandbox sessions. Sharing `_bridge` by reference would let a
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
        if new._interpreter_factory is not None and new._module_src is not None:
            from dspy.flex.bridge import BridgeRuntime

            bridge = BridgeRuntime(new, new._interpreter_factory, new._max_predictor_calls)
            bridge.bind(new._module_src)
            if self._bridge is not None:
                bridge._registry = dict(self._bridge._registry)
            new._bridge = bridge
        return new


def _find_module_class(ns: dict[str, Any]) -> type:
    """Find the generated ``dspy.Module`` subclass in an exec'd namespace."""
    candidates = [v for v in ns.values() if isinstance(v, type) and issubclass(v, Module) and v is not Module]
    defined = [c for c in candidates if "forward" in c.__dict__]
    chosen = defined or candidates
    if not chosen:
        raise RuntimeError("module_src must define a dspy.Module subclass with a `forward` method")
    return chosen[0]


_BARE_MODULE_KEYS: set[str] | None = None


def _bare_module_keys() -> set[str]:
    """Instance ``__dict__`` keys a bare ``dspy.Module`` carries; ``_bind_code`` copies only the extra ones."""
    global _BARE_MODULE_KEYS
    if _BARE_MODULE_KEYS is None:
        _BARE_MODULE_KEYS = set(Module().__dict__.keys())
    return _BARE_MODULE_KEYS
