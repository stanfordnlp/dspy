from __future__ import annotations

import hashlib
import json
import logging
import types
import warnings
from pathlib import Path
from typing import Any

import dspy
from dspy.primitives.module import Module
from dspy.utils.annotation import experimental
from dspy.vibe.codegen import VibeContext, assess_intent, repair
from dspy.vibe.persistence import PersistedVibe, parse_persisted_file, render_persisted_file

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
    code = compile(source, filename="<vibe>", mode="exec")
    globals_ns: dict[str, Any] = dict(context_names or {})
    exec(code, globals_ns)
    return globals_ns


@experimental(version="3.3.0b2")
class Vibe(Module):
    """A DSPy Module that starts as a baseline ``dspy.RLM`` and whose code is optimizable.

    Construct it like any other module — ``dspy.Vibe(MySignature)``. On first use it binds
    a deterministic baseline that simply delegates to ``dspy.RLM(<signature>)`` (no codegen
    LM call). The module is *marked* code-optimizable (``_code_optimizable``): when optimized
    with ``dspy.GEPA``, the optimizer may rewrite its source (a single ``dspy.Module`` subclass,
    exposed as ``module_src``) — decomposing the task into focused predictors and plain Python —
    rather than only tuning instructions. A regular ``dspy.Predict`` in the same program keeps
    the default instruction-only optimization.

    Args:
        signature: A ``dspy.Signature`` class (or string) declaring inputs/outputs.
        persist_to: Path to a checked-in ``.py`` file holding the implementation. When set,
            the file is loaded if its signature hash matches; otherwise a fresh baseline is
            written. ``dspy.GEPA`` writes optimized code back to this same file. When None,
            the implementation lives only in memory.
        context: Optional list of ``dspy.Tool`` / callables / style note strings to inject
            into the runtime exec namespace (and into code optimization).
        codegen_lm: LM used for auto-repair of broken code and for the intent check.
            Defaults to ``dspy.settings.lm``.
        auto_repair: When True (default), Vibe tries to recover from a broken persisted /
            optimized implementation by re-invoking the codegen LM with the broken body and
            the exception text — at most once at bind time (on load) and once at runtime per
            process. Set False to surface errors directly. The deterministic baseline never
            triggers repair.
        check_intent: When True (default), the first time a module is created for a signature
            (a fresh baseline — not a plain reload of a matching persisted file) Vibe makes one
            best-effort LM call to judge whether the signature is specific enough to implement.
            If it looks vague or misleading, Vibe emits a warning naming what's unclear and a
            question to clarify; the module is still built. Skipped silently when no LM is
            available or the check itself errors. Set False to disable the call entirely.

    Persistence: when ``persist_to`` is set, the implementation is a single self-contained
    ``.py`` file (a header comment, a signature-hash guard, the recorded Signature, and the
    generated ``dspy.Module`` subclass in a marked region). You may edit it by hand; on the next
    run a matching signature loads it as-is, while a changed signature regenerates the baseline.
    """

    # Marker read by ``dspy.GEPA``: this module's code (the ``module_src`` class) may be
    # rewritten by the optimizer, instead of only its inner predictors' instructions.
    # Duck-typed via ``getattr(obj, "_code_optimizable", False)``.
    _code_optimizable: bool = True

    def __init__(
        self,
        signature: Any,
        *,
        persist_to: str | Path | None = None,
        context: list[Any] | None = None,
        codegen_lm: dspy.LM | None = None,
        auto_repair: bool = True,
        check_intent: bool = True,
    ):
        super().__init__()

        from dspy.signatures.signature import ensure_signature

        self._signature_cls = ensure_signature(signature)
        self._name = getattr(self._signature_cls, "__name__", None) or "Vibe"
        self._persist_to: Path | None = Path(persist_to) if persist_to else None
        self._codegen_lm = codegen_lm

        style_notes: list[str] = []
        tools: list[Any] = []
        for item in context or []:
            if isinstance(item, str):
                style_notes.append(item)
            else:
                tools.append(item)
        self._vibe_ctx = VibeContext(
            signature_cls=self._signature_cls, tools=tools, style_notes=style_notes
        )

        self._module_src: str | None = None
        self._attached_names: list[str] = []
        self._forward_impl: Any = None
        self._auto_repair = auto_repair
        self._check_intent = check_intent
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
        state = super().dump_state(json_mode=json_mode)
        state["_vibe"] = {
            "module_src": self._module_src,
            "signature_hash": self._signature_hash(),
        }
        return state

    def load_state(self, state: dict[str, Any], *, allow_unsafe_lm_state: bool = False) -> None:
        vibe_state = state.pop("_vibe", None) if isinstance(state, dict) else None
        if vibe_state and vibe_state.get("module_src"):
            self._bind_code(vibe_state["module_src"])
        if state:
            super().load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)

    def _lazy_implement(self, *, force: bool = False) -> None:
        if self._module_src and not force:
            return

        sig_hash = self._signature_hash()

        if self._persist_to and self._persist_to.exists() and not force:
            stored = self._read_persisted()
            if stored is not None and stored.signature_hash == sig_hash:
                # Run exactly what's on disk (baseline, GEPA-optimized, or hand-edited).
                try:
                    self._bind_code(stored.module_src)
                except Exception as err:
                    if not self._auto_repair:
                        raise
                    fixed = self._repair_after_bind_failure(broken=stored.module_src, error=err)
                    self._write_persisted(fixed, sig_hash)
                logger.info("dspy.Vibe %r: loaded implementation from %s", self._name, self._persist_to)
                return
            if stored is not None:
                logger.info(
                    "dspy.Vibe %r: signature changed on %s — resetting to the RLM baseline.",
                    self._name,
                    self._persist_to,
                )

        # This is a fresh generation (no matching persisted impl): a new signature, a changed
        # one, or in-memory mode. Pre-flight the signature for clarity before building.
        self._maybe_warn_intent()

        # Bind the deterministic RLM baseline (no codegen LM call). Decomposition into a
        # richer, mostly-Python program happens later via dspy.GEPA.
        module_src = self._rlm_baseline_src()
        self._bind_code(module_src)
        if self._persist_to is not None:
            self._write_persisted(module_src, sig_hash)
            logger.info("dspy.Vibe %r: wrote RLM baseline to %s", self._name, self._persist_to)
        else:
            logger.info("dspy.Vibe %r: persist_to=None — RLM baseline is in-memory only.", self._name)

    def _maybe_warn_intent(self) -> None:
        """Best-effort: warn if the signature looks too vague/misleading to implement well.

        Runs at most one LM call, only on a fresh generation (the caller gates this to the
        fresh-baseline path, never a plain reload). Skipped when ``check_intent`` is off or no
        LM is available, and any failure is swallowed — an advisory check must never block
        construction. The module is built regardless of the verdict.
        """
        if not self._check_intent:
            return
        lm = self._codegen_lm or dspy.settings.lm
        if lm is None:
            return
        try:
            is_clear, vague_aspect, clarifying_question = assess_intent(self._vibe_ctx, lm=lm)
        except Exception as err:  # advisory only — never block construction
            logger.debug("dspy.Vibe %r: intent check skipped (%s)", self._name, err)
            return
        if is_clear:
            return
        message = (
            f"dspy.Vibe {self._name!r}: this signature may be too vague or misleading to "
            f"implement reliably, so the generated module may be incorrect.\n"
            f"  What's unclear: {vague_aspect or '(the objective and/or field roles are underspecified)'}\n"
            f"  Please clarify: {clarifying_question or 'tighten the objective (docstring) and the input/output field descriptions.'}\n"
            f"  (Pass check_intent=False to dspy.Vibe(...) to silence this check.)"
        )
        logger.warning(message)
        warnings.warn(message, stacklevel=2)

    def _rlm_baseline_src(self) -> str:
        """Deterministic baseline source: a ``dspy.Module`` that delegates to one ``dspy.RLM``.

        No LM call — emits a class whose ``__init__`` constructs ``dspy.RLM(<signature>)`` and
        whose ``forward`` unwraps its declared outputs. dspy.GEPA later rewrites this into a
        decomposed, mostly-Python module. Carries the signature's instructions into the RLM so
        the baseline sees the full task description; references only ``dspy`` so the persisted
        file stays self-contained (``_bind_code`` execs with only ``dspy`` and context tools in
        scope).
        """
        cls: Any = self._signature_cls
        sig_str = self._vibe_ctx.render_signature_string()
        output_names = list(cls.output_fields.keys())
        instructions = (getattr(cls, "instructions", "") or "").strip()
        # `!r` escapes quotes/newlines into a valid literal, and the rebuilt signature only
        # references `dspy`, so the persisted file stays self-contained.
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
        # Match the normalized (stripped) form the persistence round-trip yields, so a freshly
        # written baseline isn't misread as a hand edit on its first reload.
        return src.strip()

    def _class_name(self) -> str:
        """Name for the generated module class, derived from the signature name."""
        base = self._name if (self._name and self._name.isidentifier()) else "Vibe"
        return f"{base}Module"

    def _repair_after_bind_failure(self, *, broken: str, error: BaseException) -> str:
        """Invoke the repair codegen LM after ``_bind_code`` raised, and bind the fix.

        The bind of the *repaired* code is not wrapped — if the repair itself is broken, the
        user sees that as a hard error rather than an infinite repair loop. Returns the fixed
        ``module_src``; the caller persists it.
        """
        error_text = f"{type(error).__name__}: {error}"
        logger.warning(
            "dspy.Vibe %r: bind failure (%s) — running auto-repair. "
            "Set auto_repair=False to surface the error directly.",
            self._name,
            error_text,
        )
        fixed = repair(
            self._vibe_ctx, broken=broken, failure_kind="bind", error_text=error_text, lm=self._codegen_lm
        )
        self._bind_code(fixed)
        return fixed

    def _bind_code(self, module_src: str) -> None:
        """Exec ``module_src`` (a ``dspy.Module`` subclass) and attach its predictors onto ``self``.

        The generated/persisted implementation is a normal ``dspy.Module`` subclass. We exec it,
        instantiate it once (LM-free — only predictor *construction* runs), and copy whatever its
        ``__init__`` defined (predictors and any plain attributes) onto this Vibe instance, then
        bind its ``forward`` as ``self._forward_impl``. So ``self`` behaves exactly like the
        generated module — ``self.<predictor>`` and ``named_predictors()`` stay flat — while the
        source remains a clean, editable class.
        """
        ctx_names = self._vibe_ctx.context_names()
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
        and ``auto_repair`` is on, Vibe re-invokes the codegen LM with the broken body + the
        exception, swaps in the fix, and re-runs the call once. Subsequent failures (or
        non-repairable exceptions like a tool/LM error) propagate unchanged.
        """
        if self._forward_impl is None:
            raise RuntimeError(
                f"dspy.Vibe {self._name!r}: no implementation bound. Did __init__ or implement() fail?"
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
        """Repair, rebind, and re-persist after a runtime exception in forward().

        Driven by an exception raised while ``_forward_impl`` was running; the broken
        candidate is the currently-bound code. If repair codegen produces another
        bind-broken body, the bind error propagates — we do not loop on repair.
        """
        assert self._module_src is not None
        broken = self._module_src
        error_text = f"{type(error).__name__}: {error}"
        logger.warning(
            "dspy.Vibe %r: runtime failure (%s) — running auto-repair. "
            "Set auto_repair=False to surface the error directly.",
            self._name,
            error_text,
        )
        fixed = repair(
            self._vibe_ctx, broken=broken, failure_kind="runtime", error_text=error_text, lm=self._codegen_lm
        )
        self._bind_code(fixed)
        if self._persist_to is not None:
            self._write_persisted(fixed, self._signature_hash())

    def _read_persisted(self) -> PersistedVibe | None:
        if self._persist_to is None or not self._persist_to.exists():
            return None
        parsed = parse_persisted_file(self._persist_to.read_text(encoding="utf-8"))
        if parsed is None:
            logger.warning(
                "dspy.Vibe %r: persisted file at %s is missing expected markers",
                self._name,
                self._persist_to,
            )
        return parsed

    def _write_persisted(self, module_src: str, sig_hash: str) -> None:
        assert self._persist_to is not None
        self._persist_to.parent.mkdir(parents=True, exist_ok=True)
        body = render_persisted_file(
            signature_hash=sig_hash,
            signature_name=self._name,
            module_src=module_src,
            signature_spec=self._vibe_ctx.render_signature_spec(),
        )
        self._persist_to.write_text(body, encoding="utf-8")

    def _signature_hash(self) -> str:
        cls: Any = self._signature_cls
        canonical = {
            "name": getattr(cls, "__name__", ""),
            "instructions": getattr(cls, "instructions", "") or "",
            "input_fields": [(n, _annotation_repr(f.annotation)) for n, f in cls.input_fields.items()],
            "output_fields": [(n, _annotation_repr(f.annotation)) for n, f in cls.output_fields.items()],
        }
        blob = json.dumps(canonical, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def _annotation_repr(t: Any) -> str:
    if t is None:
        return "None"
    name = getattr(t, "__name__", None)
    if name:
        return name
    return str(t).replace("typing.", "")


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
