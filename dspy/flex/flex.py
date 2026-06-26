from __future__ import annotations

import logging
import types
from typing import Any

import dspy
from dspy.flex.ctx import FlexContext, repair
from dspy.primitives.module import Module
from dspy.utils.annotation import experimental

# Runtime exceptions that we treat as "the user's code is wrong, not a flaky
# downstream call." A bare LM or tool call typically raises a more specific
# subclass that doesn't appear here (e.g. anthropic.APIError, openai.OpenAIError).
_RUNTIME_REPAIRABLE_ERRORS: tuple[type[BaseException], ...] = (
    AttributeError,
    TypeError,
    NameError,
    KeyError,
)

logger = logging.getLogger(__name__)


def _exec_source(source: str, context_names: dict[str, Any] | None = None) -> dict[str, Any]:
    code = compile(source, filename="<flex>", mode="exec")
    globals_ns: dict[str, Any] = dict(context_names or {})
    exec(code, globals_ns)
    return globals_ns


@experimental(version="3.3.0b2")
class Flex(Module):
    """A DSPy Module that starts as a baseline ``dspy.RLM`` and whose code is optimizable.

    Construct it like any other module — ``dspy.Flex(MySignature)``. On first use it binds
    a deterministic baseline that simply delegates to ``dspy.RLM(<signature>)`` (no codegen
    LM call). The module is *marked* code-optimizable (``_code_optimizable``): when optimized
    with ``dspy.GEPA``, the optimizer may rewrite its source (a single ``dspy.Module`` subclass,
    exposed as ``module_src``) — decomposing the task into predictors and code —
    rather than only tuning instructions.

    Persistence uses the standard ``dspy.Module`` API. ``save`` / ``load`` (and ``dspy.load``)
    round-trip the generated ``module_src`` together with the inner predictors' state, exactly
    as instruction-optimized modules are saved — there is no separate on-disk format::

        optimized = dspy.GEPA(...).compile(dspy.Flex(MySignature), trainset=...)
        optimized.save("program.json")
        # later:
        program = dspy.Flex(MySignature)
        program.load("program.json")   # rebinds the optimized code, then its predictor state

    Args:
        signature: A ``dspy.Signature`` class (or string) declaring inputs/outputs.
        context: Optional list of ``dspy.Tool`` / callables / style note strings to inject
            into the runtime exec namespace (and into code optimization).
        codegen_lm: LM used for auto-repair of broken code. Defaults to ``dspy.settings.lm``.
        auto_repair: When True (default), Flex tries to recover from a broken loaded /
            optimized implementation by re-invoking the codegen LM with the broken body and
            the exception text — at most once at bind time (on load) and once at runtime per
            process. Set False to surface errors directly. The deterministic baseline never
            triggers repair.
    """

    # Marker read by ``dspy.GEPA``: this module's code (the ``module_src`` class) may be
    # rewritten by the optimizer, instead of only its inner predictors' instructions.
    # Duck-typed via ``getattr(obj, "_code_optimizable", False)``.
    _code_optimizable: bool = True

    def __init__(
        self,
        signature: Any,
        *,
        context: list[Any] | None = None,
        codegen_lm: dspy.LM | None = None,
        auto_repair: bool = True,
    ):
        super().__init__()

        from dspy.signatures.signature import ensure_signature

        self._signature_cls = ensure_signature(signature)
        self._name = getattr(self._signature_cls, "__name__", None) or "Flex"
        self._codegen_lm = codegen_lm

        style_notes: list[str] = []
        tools: list[Any] = []
        for item in context or []:
            if isinstance(item, str):
                style_notes.append(item)
            else:
                tools.append(item)
        self._flex_ctx = FlexContext(
            signature_cls=self._signature_cls, tools=tools, style_notes=style_notes
        )

        self._module_src: str | None = None
        self._attached_names: list[str] = []
        self._forward_impl: Any = None
        self._auto_repair = auto_repair
        self._runtime_repair_used = False

        self._lazy_implement()

    def implement(self, *, force: bool = False) -> None:
        """Explicitly (re-)generate the implementation."""
        if force:
            self._module_src = None
        self._lazy_implement(force=force)

    @property
    def signature(self) -> Any:
        return self._signature_cls

    @property
    def module_src(self) -> str | None:
        """The generated implementation: source of a single ``dspy.Module`` subclass."""
        return self._module_src

    def dump_state(self, json_mode: bool = True) -> dict[str, Any]:
        # The generated code rides alongside the predictors' state, so a single
        # Module.save captures both the architecture (module_src) and the tuned predictors.
        state = super().dump_state(json_mode=json_mode)
        state["_flex"] = {"module_src": self._module_src}
        return state

    def load_state(self, state: dict[str, Any], *, allow_unsafe_lm_state: bool = False) -> None:
        # Rebind the saved code FIRST (it defines which predictors exist), then let the base
        # Module load each predictor's state onto the freshly-bound predictors.
        flex_state = state.pop("_flex", None) if isinstance(state, dict) else None
        if flex_state and flex_state.get("module_src"):
            self._bind_with_repair(flex_state["module_src"])
        if state:
            super().load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)

    def _lazy_implement(self, *, force: bool = False) -> None:
        if self._module_src and not force:
            return
        # Bind the deterministic RLM baseline (no codegen LM call). Decomposition into a
        # richer, mostly-Python program happens later via dspy.GEPA; the optimized code is
        # carried across processes by Module.save / load (see dump_state / load_state).
        self._bind_code(self._rlm_baseline_src())

    def _rlm_baseline_src(self) -> str:
        """Deterministic baseline source: a ``dspy.Module`` that delegates to one ``dspy.RLM``.

        No LM call — emits a class whose ``__init__`` constructs ``dspy.RLM(<signature>)`` and
        whose ``forward`` unwraps its declared outputs. dspy.GEPA later rewrites this into a
        decomposed, mostly-Python module. Carries the signature's instructions into the RLM so
        the baseline sees the full task description; references only ``dspy`` so the bound
        module stays self-contained (``_bind_code`` execs with only ``dspy`` and context tools
        in scope).
        """
        cls: Any = self._signature_cls
        sig_str = self._flex_ctx.render_signature_string()
        output_names = list(cls.output_fields.keys())
        instructions = (getattr(cls, "instructions", "") or "").strip()
        # `!r` escapes quotes/newlines into a valid literal, and the rebuilt signature only
        # references `dspy`, so the source stays self-contained.
        rlm_arg = f"dspy.Signature({sig_str!r}, {instructions!r})" if instructions else repr(sig_str)
        returns = ", ".join(f"{name}=result.{name}" for name in output_names)
        src = (
            f"class {self._class_name()}(dspy.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            f"        self.rlm = dspy.RLM({rlm_arg})\n"
            "\n"
            "    def forward(self, **inputs):\n"
            "        result = self.rlm(**inputs)\n"
            f"        return dspy.Prediction({returns})"
        )
        return src.strip()

    def _class_name(self) -> str:
        """Name for the generated module class, derived from the signature name."""
        base = self._name if (self._name and self._name.isidentifier()) else "Flex"
        return f"{base}Module"

    def _bind_with_repair(self, module_src: str) -> None:
        """Bind ``module_src``, auto-repairing once if the bind fails (when ``auto_repair`` is on).

        Used on the load path: a saved/optimized — or hand-edited — implementation that no
        longer execs cleanly is handed to the codegen LM to fix, mirroring the runtime repair
        in ``forward``. With ``auto_repair=False`` the bind error surfaces directly.
        """
        try:
            self._bind_code(module_src)
        except Exception as err:
            if not self._auto_repair:
                raise
            self._repair_after_bind_failure(broken=module_src, error=err)

    def _repair_after_bind_failure(self, *, broken: str, error: BaseException) -> str:
        """Invoke the repair codegen LM after ``_bind_code`` raised, and bind the fix.

        The bind of the *repaired* code is not wrapped — if the repair itself is broken, the
        user sees that as a hard error rather than an infinite repair loop. Returns the fixed
        ``module_src``.
        """
        error_text = f"{type(error).__name__}: {error}"
        logger.warning(
            "dspy.Flex %r: bind failure (%s) — running auto-repair. "
            "Set auto_repair=False to surface the error directly.",
            self._name,
            error_text,
        )
        fixed = repair(
            self._flex_ctx, broken=broken, failure_kind="bind", error_text=error_text, lm=self._codegen_lm
        )
        self._bind_code(fixed)
        return fixed

    def _bind_code(self, module_src: str) -> None:
        """Exec ``module_src`` (a ``dspy.Module`` subclass) and attach its predictors onto ``self``.

        The generated implementation is a normal ``dspy.Module`` subclass. We exec it,
        instantiate it once (LM-free — only predictor *construction* runs), and copy whatever its
        ``__init__`` defined (predictors and any plain attributes) onto this Flex instance, then
        bind its ``forward`` as ``self._forward_impl``. So ``self`` behaves exactly like the
        generated module — ``self.<predictor>`` and ``named_predictors()`` stay flat — while the
        source remains a clean, editable class.
        """
        ctx_names = self._flex_ctx.context_names()
        ctx_names["dspy"] = dspy

        # Detach attributes from the previous binding before re-attaching.
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

        # Bind the generated forward to `self` (its __globals__ already has `dspy` + tools).
        # The class-level ``forward`` wraps this with runtime auto-repair.
        self._forward_impl = types.MethodType(forward_fn, self)
        self._module_src = module_src

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the LM-authored ``_forward_impl`` with runtime auto-repair.

        If a call raises one of the narrow user-code errors in ``_RUNTIME_REPAIRABLE_ERRORS``
        and ``auto_repair`` is on, Flex re-invokes the codegen LM with the broken body + the
        exception, swaps in the fix, and re-runs the call once. Subsequent failures (or
        non-repairable exceptions like a tool/LM error) propagate unchanged.
        """
        if self._forward_impl is None:
            raise RuntimeError(
                f"dspy.Flex {self._name!r}: no implementation bound. Did __init__ or implement() fail?"
            )
        try:
            return self._forward_impl(*args, **kwargs)
        except _RUNTIME_REPAIRABLE_ERRORS as err:
            if not self._auto_repair or self._runtime_repair_used:
                raise
            self._runtime_repair_used = True
            self._repair_after_runtime_failure(error=err)
            return self._forward_impl(*args, **kwargs)

    def _repair_after_runtime_failure(self, *, error: BaseException) -> None:
        """Repair and rebind after a runtime exception in forward().

        Driven by an exception raised while ``_forward_impl`` was running; the broken
        candidate is the currently-bound code. If repair codegen produces another
        bind-broken body, the bind error propagates — we do not loop on repair. The fix lives
        in memory; persist it with ``save`` if you want to keep it.
        """
        assert self._module_src is not None
        broken = self._module_src
        error_text = f"{type(error).__name__}: {error}"
        logger.warning(
            "dspy.Flex %r: runtime failure (%s) — running auto-repair. "
            "Set auto_repair=False to surface the error directly.",
            self._name,
            error_text,
        )
        fixed = repair(
            self._flex_ctx, broken=broken, failure_kind="runtime", error_text=error_text, lm=self._codegen_lm
        )
        self._bind_code(fixed)


def _find_module_class(ns: dict[str, Any]) -> type:
    """Find the generated ``dspy.Module`` subclass in an exec'd namespace.

    Prefers a class that defines its own ``forward`` (the implementation), so an imported or
    helper base class doesn't win. Raises if none is present.
    """
    candidates = [v for v in ns.values() if isinstance(v, type) and issubclass(v, Module) and v is not Module]
    defined = [c for c in candidates if "forward" in c.__dict__]
    chosen = defined or candidates
    if not chosen:
        raise RuntimeError("module_src must define a dspy.Module subclass with a `forward` method")
    return chosen[0]


_BARE_MODULE_KEYS: set[str] | None = None


def _bare_module_keys() -> set[str]:
    """Instance ``__dict__`` keys a bare ``dspy.Module`` carries (its internal bookkeeping).

    Computed once and cached. ``_bind_code`` copies only the *extra* keys a generated
    ``__init__`` adds (the predictors and any plain state), never these internals.
    """
    global _BARE_MODULE_KEYS
    if _BARE_MODULE_KEYS is None:
        _BARE_MODULE_KEYS = set(Module().__dict__.keys())
    return _BARE_MODULE_KEYS
