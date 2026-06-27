from __future__ import annotations

import types
from typing import Any

import dspy
from dspy.flex.ctx import FlexContext
from dspy.primitives.module import Module
from dspy.utils.annotation import experimental


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

    Args:
        signature: A ``dspy.Signature`` class (or string) declaring inputs/outputs.
        tools: ``dspy.Tool`` instances or named callables.
    """

    # Read by ``dspy.GEPA`` (duck-typed): this module's code may be rewritten by the optimizer.
    _code_optimizable: bool = True

    def __init__(
        self,
        signature: Any,
        *,
        tools: list[Any] | None = None,
    ):
        super().__init__()

        from dspy.signatures.signature import ensure_signature

        self._signature_cls = ensure_signature(signature)
        self._name = getattr(self._signature_cls, "__name__", None) or "Flex"
        self._flex_ctx = FlexContext(signature_cls=self._signature_cls, tools=list(tools or []))

        self._module_src: str | None = None
        self._attached_names: list[str] = []
        self._forward_impl: Any = None

        # Bind the deterministic RLM baseline (no LM call); GEPA may later rewrite this code.
        self._bind_code(self._rlm_baseline_src())

    @property
    def signature(self) -> Any:
        return self._signature_cls

    @property
    def module_src(self) -> str | None:
        """Source of the bound implementation (a single ``dspy.Module`` subclass)."""
        return self._module_src

    def dump_state(self, json_mode: bool = True) -> dict[str, Any]:
        state = super().dump_state(json_mode=json_mode)
        state["_flex"] = {"module_src": self._module_src}
        return state

    def load_state(self, state: dict[str, Any], *, allow_unsafe_lm_state: bool = False) -> None:
        flex_state = state.pop("_flex", None) if isinstance(state, dict) else None
        if flex_state and flex_state.get("module_src"):
            self._bind_code(flex_state["module_src"])
        if state:
            super().load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)

    def _rlm_baseline_src(self) -> str:
        """Baseline source: a ``dspy.Module`` delegating to one ``dspy.RLM``."""
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
        """Exec ``module_src`` and attach the resulting module's predictors onto ``self``."""
        ctx_names = self._flex_ctx.context_names()
        ctx_names["dspy"] = dspy

        for old_name in self._attached_names:
            if hasattr(self, old_name):
                delattr(self, old_name)
        self._attached_names = []

        ns = _exec_source(module_src, context_names=ctx_names)
        impl_cls = _find_module_class(ns)
        forward_fn = impl_cls.__dict__.get("forward")
        if not callable(forward_fn):
            raise RuntimeError("module_src's dspy.Module subclass must define a `forward` method")

        impl = impl_cls()  # LM-free: runs __init__, constructing predictors only
        baseline_keys = _bare_module_keys()
        for key, value in list(impl.__dict__.items()):
            if key in baseline_keys:
                continue  # internal dspy.Module bookkeeping, not user-defined state
            setattr(self, key, value)
            self._attached_names.append(key)

        self._forward_impl = types.MethodType(forward_fn, self)
        self._module_src = module_src

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the bound ``forward``."""
        if self._forward_impl is None:
            raise RuntimeError(f"dspy.Flex {self._name!r}: no implementation bound.")
        return self._forward_impl(*args, **kwargs)


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
