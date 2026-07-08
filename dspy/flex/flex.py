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
    """A DSPy Module that starts as a ``dspy.RLM`` baseline and whose code is optimizable.

    Construct it like any module — ``dspy.Flex(MySignature)`` — and on first use it binds a
    deterministic baseline delegating to ``dspy.RLM(<signature>)``. It is marked
    ``_code_optimizable``: ``dspy.GEPA`` may rewrite its source (a single ``dspy.Module``
    subclass, exposed as ``module_src``) into decomposed predictors + code, rather than only
    tuning instructions.

    Sandboxing (``interpreter``):
        Pass ``interpreter`` to run **all** generated code inside a sandbox (a ``CodeInterpreter`` such
        as ``dspy.PythonInterpreter``) instead of in-process ``exec``. The optimizer-authored glue —
        ``__init__``/``forward`` bodies, including imports — runs isolated with no filesystem/network
        beyond what the interpreter enables; only predictor construction and predictor calls bridge
        back to the host, which performs the real LM calls. This holds even after ``dspy.GEPA`` rewrites
        the module into a plain ``ChainOfThought``/``Predict``, so the sandbox cannot be optimized away.
        It does **not** sandbox the host LM calls themselves; treat ``module_src`` with the trust you
        place in the optimizer/LM that authored it for cost/serialization concerns. ``max_predictor_calls``
        bounds how many LM calls one sandboxed ``forward`` may drive (runaway/abuse guard).

        Provide a zero-arg **factory** (``interpreter=lambda: dspy.PythonInterpreter(...)``) for
        isolation under parallel evaluation; a bare instance is shared and warned about. The interpreter
        is a runtime dependency (like the LM) and is not serialized; re-supply it when reconstructing a
        Flex before ``load``.

        Backend choice: the bridge is **interpreter-agnostic** — it depends only on the
        ``CodeInterpreter`` protocol (host tools callable from sandbox code), never on the default
        Deno/Pyodide internals. ``dspy.PythonInterpreter`` (Deno + Pyodide WASM) is the zero-infra
        default and is well suited to the *bounded* orchestration glue Flex runs. For stronger,
        full-CPython isolation, plug in any ``CodeInterpreter`` backed by a **local microVM** (e.g.
        microsandbox / libkrun) or gVisor; the bridge needs the sandbox to call back to the host, so a
        *local* sandbox fits, whereas a *remote/hosted* one (e.g. cloud E2B) would need a callback
        tunnel. Code-executing sub-predictors a rewrite may introduce (``RLM``/``CodeAct``/
        ``ProgramOfThought``) run their own inner code in a fresh interpreter from this same factory,
        so that code runs in the sandbox backend you chose here rather than a separate default one
        (unless ``interpreter`` was a shared instance, in which case they fall back to their own
        default sandbox — pass a factory to have them inherit the configured backend).

    Args:
        signature: A ``dspy.Signature`` class (or string) declaring inputs/outputs.
        tools: ``dspy.Tool`` instances or named callables.
        interpreter: Optional ``CodeInterpreter`` instance or zero-arg factory. When set, generated code
            runs inside the sandbox via the run-in-interpreter bridge (see ``dspy/flex/bridge.py``).
        max_predictor_calls: Max bridged predictor (LM) calls allowed per sandboxed ``forward``; raises
            past the cap. ``None`` disables the cap. Ignored when no ``interpreter`` is set.
    """

    # Read by ``dspy.GEPA`` (duck-typed): this module's code may be rewritten by the optimizer.
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

        # Optional sandbox. When set, generated code runs *inside* `interpreter` via the run-in-
        # interpreter bridge (dspy/flex/bridge.py) rather than in-process exec(); see the class
        # docstring for what this does and does not protect.
        self._interpreter_factory = self._normalize_interpreter(interpreter)
        # A bare CodeInterpreter instance is shared across all forward() calls (see
        # _normalize_interpreter). A code-executing sub-predictor (CodeAct/ProgramOfThought/RLM) shuts
        # its interpreter down after forward, so it must NOT be handed the shared Flex session — only a
        # real factory (fresh instances) is safe to inherit into sub-predictors. See
        # BridgeRuntime._sub_interpreter.
        self._interpreter_shared = isinstance(interpreter, CodeInterpreter)
        self._max_predictor_calls = max_predictor_calls
        self._bridge: Any = None
        if self._interpreter_factory is not None:
            from dspy.flex.bridge import BridgeRuntime

            self._bridge = BridgeRuntime(self, self._interpreter_factory, self._max_predictor_calls)

        self._bind_code(self._rlm_baseline_src())

    @staticmethod
    def _normalize_interpreter(
        interpreter: CodeInterpreter | Callable[[], CodeInterpreter] | None,
    ) -> Callable[[], CodeInterpreter] | None:
        """Normalize an interpreter instance-or-factory to a zero-arg factory (or ``None``)."""
        if interpreter is None:
            return None
        # Order matters: a CodeInterpreter instance (e.g. MockInterpreter) may also be callable, so we
        # must classify it as an instance first and not mistake it for a factory.
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
        module_src = state.pop("module_src", None) if isinstance(state, dict) else None
        if module_src:
            self._bind_code(module_src)
        if state:
            super().load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)

    def _rlm_baseline_src(self) -> str:
        cls: Any = self._signature_cls
        sig_str = self._flex_ctx.render_signature_string()
        returns = ", ".join(f"{name}=result.{name}" for name in cls.output_fields)
        instructions = (getattr(cls, "instructions", "") or "").strip()
        rlm_arg = f"dspy.Signature({sig_str!r}, {instructions!r})" if instructions else repr(sig_str)
        tool_names = list(self._flex_ctx.context_names())
        tools_arg = f", tools=[{', '.join(tool_names)}]" if tool_names else ""
        return (
            f"class {self._class_name()}(dspy.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.rlm = dspy.RLM({rlm_arg}{tools_arg})\n"
            "\n"
            "    def forward(self, **inputs):\n"
            "        result = self.rlm(**inputs)\n"
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
        """Bind ``module_src`` for sandboxed execution.

        The host never ``exec``s the optimizer-authored source. Predictors are constructed *inside*
        the sandbox, which bridges construction back to the host (attaching them onto ``self``), and
        ``forward`` runs in the sandbox too. We instantiate eagerly here — like the in-process path's
        ``__init__`` — so ``named_predictors()``/``load_state``/serialization see the predictors.
        """
        self._forward_impl = None  # forward is handled by the bridge, not an in-process method
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
        # Each copy must own its sandbox sessions. A reference-shared `_bridge` (deepcopy can't copy
        # the live interpreters) would let a throwaway copy's __del__ shut down the original's session
        # — e.g. the trial copy inside Module.load_state. So deep-copy everything except `_bridge` and
        # give the copy its own session-less BridgeRuntime, seeded to reuse the already-built
        # predictors (which were deep-copied with their state) instead of rebuilding them.
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
