from __future__ import annotations

import hashlib
import json
import logging
import types
from pathlib import Path
from typing import Any

import dspy
from dspy.primitives.module import Module
from dspy.utils.annotation import experimental
from dspy.vibe.codegen import repair
from dspy.vibe.persistence import parse_persisted_file, render_persisted_file

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
    with ``dspy.GEPA``, the optimizer may rewrite its source (``predictors_src`` /
    ``forward_src``) — decomposing the task into focused predictors and plain Python — rather
    than only tuning instructions. A regular ``dspy.Predict`` in the same program keeps the
    default instruction-only optimization.

    Args:
        signature: A ``dspy.Signature`` class (or string) declaring inputs/outputs.
        persist_to: Path to a checked-in ``.py`` file holding the implementation. When set,
            the file is loaded if its signature hash matches; otherwise a fresh baseline is
            written. ``dspy.GEPA`` writes optimized code back to this same file. When None,
            the implementation lives only in memory.
        context: Optional list of ``dspy.Tool`` / callables / style note strings to inject
            into the runtime exec namespace (and into code optimization).
        codegen_lm: LM used for auto-repair of broken code. Defaults to ``dspy.settings.lm``.
        auto_repair: When True (default), Vibe tries to recover from a broken persisted /
            optimized implementation by re-invoking the codegen LM with the broken body and
            the exception text — at most once at bind time (on load) and once at runtime per
            process. Set False to surface errors directly. The deterministic baseline never
            triggers repair.

    Persistence: when ``persist_to`` is set, the implementation is a single self-contained
    ``.py`` file (a header comment, a signature-hash guard, and the ``PREDICTORS`` dict +
    ``forward`` function in marked regions). You may edit it by hand; on the next run a
    matching signature loads it as-is, while a changed signature regenerates the baseline.
    """

    # Marker read by ``dspy.GEPA``: this module's code (``predictors_src`` / ``forward_src``)
    # may be rewritten by the optimizer, instead of only its inner predictors' instructions.
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

        self._forward_src: str | None = None
        self._predictors_src: str | None = None
        self._predictor_names: list[str] = []
        self._forward_impl: Any = None
        self._auto_repair = auto_repair
        self._runtime_repair_used = False

        self._lazy_implement()

    def implement(self, *, force: bool = False) -> None:
        """Explicitly (re-)generate the implementation."""
        if force:
            self._predictors_src = None
            self._forward_src = None
        self._lazy_implement(force=force)

    @property
    def signature(self) -> Any:
        return self._signature_cls

    @property
    def predictors_src(self) -> str | None:
        return self._predictors_src

    @property
    def forward_src(self) -> str | None:
        return self._forward_src

    def dump_state(self, json_mode: bool = True) -> dict[str, Any]:
        state = super().dump_state(json_mode=json_mode)
        state["_vibe"] = {
            "forward_src": self._forward_src,
            "predictors_src": self._predictors_src,
            "signature_hash": self._signature_hash(),
        }
        return state

    def load_state(self, state: dict[str, Any], *, allow_unsafe_lm_state: bool = False) -> None:
        vibe_state = state.pop("_vibe", None) if isinstance(state, dict) else None
        if vibe_state and vibe_state.get("forward_src") and vibe_state.get("predictors_src"):
            self._bind_code(vibe_state["predictors_src"], vibe_state["forward_src"])
        if state:
            super().load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)

    def _lazy_implement(self, *, force: bool = False) -> None:
        if self._forward_src and self._predictors_src and not force:
            return

        sig_hash = self._signature_hash()

        if self._persist_to and self._persist_to.exists() and not force:
            stored = self._read_persisted()
            if stored is not None and stored.signature_hash == sig_hash:
                # Run exactly what's on disk (baseline, GEPA-optimized, or hand-edited).
                try:
                    self._bind_code(stored.predictors_src, stored.forward_src)
                except Exception as err:
                    if not self._auto_repair:
                        raise
                    fixed_p, fixed_f = self._repair_after_bind_failure(
                        broken=(stored.predictors_src, stored.forward_src), error=err
                    )
                    self._write_persisted(fixed_p, fixed_f, sig_hash)
                logger.info("dspy.Vibe %r: loaded implementation from %s", self._name, self._persist_to)
                return
            if stored is not None:
                logger.info(
                    "dspy.Vibe %r: signature changed on %s — resetting to the RLM baseline.",
                    self._name,
                    self._persist_to,
                )

        # Bind the deterministic RLM baseline (no codegen LM call). Decomposition into a
        # richer, mostly-Python program happens later via dspy.GEPA.
        predictors_src, forward_src = self._rlm_baseline_src()
        self._bind_code(predictors_src, forward_src)
        if self._persist_to is not None:
            self._write_persisted(predictors_src, forward_src, sig_hash)
            logger.info("dspy.Vibe %r: wrote RLM baseline to %s", self._name, self._persist_to)
        else:
            logger.info("dspy.Vibe %r: persist_to=None — RLM baseline is in-memory only.", self._name)

    def _rlm_baseline_src(self) -> tuple[str, str]:
        """Deterministic baseline source: delegate the whole signature to one dspy.RLM.

        No LM call — emits source that constructs ``dspy.RLM(<signature>)`` and unwraps its
        declared outputs. dspy.GEPA later rewrites this into a decomposed, mostly-Python
        implementation. Carries the signature's instructions into the RLM so the baseline
        sees the full task description; references only ``dspy`` so the persisted file stays
        self-contained (``_bind_code`` execs with only ``dspy`` and context tools in scope).
        """
        cls: Any = self._signature_cls
        sig_str = self._vibe_ctx.render_signature_string()
        output_names = list(cls.output_fields.keys())
        instructions = (getattr(cls, "instructions", "") or "").strip()
        # `!r` escapes quotes/newlines into a valid literal, and the rebuilt signature only
        # references `dspy`, so the persisted file stays self-contained.
        rlm_arg = f"dspy.Signature({sig_str!r}, {instructions!r})" if instructions else repr(sig_str)
        predictors_src = "PREDICTORS = {\n" f"    {'rlm'!r}: dspy.RLM({rlm_arg}),\n" "}"
        returns = ", ".join(f"{name}=result.{name}" for name in output_names)
        forward_src = (
            "def forward(self, **inputs):\n"
            "    result = self.rlm(**inputs)\n"
            f"    return dspy.Prediction({returns})"
        )
        # Match the normalized (stripped) form the persistence round-trip yields, so a freshly
        # written baseline isn't misread as a hand edit on its first reload.
        return predictors_src.strip(), forward_src.strip()

    def _repair_after_bind_failure(
        self, *, broken: tuple[str, str], error: BaseException
    ) -> tuple[str, str]:
        """Invoke the repair codegen LM after ``_bind_code`` raised, and bind the fix.

        The bind of the *repaired* code is not wrapped — if the repair itself is broken, the
        user sees that as a hard error rather than an infinite repair loop. Returns the fixed
        ``(predictors_src, forward_src)``; the caller persists them.
        """
        error_text = f"{type(error).__name__}: {error}"
        logger.warning(
            "dspy.Vibe %r: bind failure (%s) — running auto-repair. "
            "Set auto_repair=False to surface the error directly.",
            self._name,
            error_text,
        )
        fixed_predictors, fixed_forward = repair(
            self._vibe_ctx, broken=broken, failure_kind="bind", error_text=error_text, lm=self._codegen_lm
        )
        self._bind_code(fixed_predictors, fixed_forward)
        return fixed_predictors, fixed_forward

    def _bind_code(self, predictors_src: str, forward_src: str) -> None:
        """Exec and attach both source artifacts to ``self``."""
        ctx_names = self._vibe_ctx.context_names()
        ctx_names["dspy"] = dspy

        for old_name in self._predictor_names:
            if hasattr(self, old_name):
                delattr(self, old_name)
        self._predictor_names = []

        pred_ns = _exec_source(predictors_src, context_names=ctx_names)
        predictors = pred_ns.get("PREDICTORS")
        if not isinstance(predictors, dict):
            raise RuntimeError("predictors_src must define a `PREDICTORS = {...}` dict at module scope")
        for name, predictor in predictors.items():
            if not isinstance(name, str) or not name.isidentifier():
                raise RuntimeError(f"PREDICTORS key {name!r} is not a valid Python identifier")
            setattr(self, name, predictor)
            self._predictor_names.append(name)

        fwd_ns = _exec_source(forward_src, context_names=ctx_names)
        forward_fn = fwd_ns.get("forward")
        if not callable(forward_fn):
            raise RuntimeError("forward_src must define a `def forward(self, ...)` function")
        # Bind to `_forward_impl` instead of `forward`. The class-level ``forward`` wraps
        # this with runtime auto-repair.
        self._forward_impl = types.MethodType(forward_fn, self)

        self._predictors_src = predictors_src
        self._forward_src = forward_src

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
        assert self._predictors_src is not None and self._forward_src is not None
        broken = (self._predictors_src, self._forward_src)
        error_text = f"{type(error).__name__}: {error}"
        logger.warning(
            "dspy.Vibe %r: runtime failure (%s) — running auto-repair. "
            "Set auto_repair=False to surface the error directly.",
            self._name,
            error_text,
        )
        fixed_predictors, fixed_forward = repair(
            self._vibe_ctx, broken=broken, failure_kind="runtime", error_text=error_text, lm=self._codegen_lm
        )
        self._bind_code(fixed_predictors, fixed_forward)
        if self._persist_to is not None:
            self._write_persisted(fixed_predictors, fixed_forward, self._signature_hash())

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

    def _write_persisted(self, predictors_src: str, forward_src: str, sig_hash: str) -> None:
        assert self._persist_to is not None
        self._persist_to.parent.mkdir(parents=True, exist_ok=True)
        body = render_persisted_file(
            signature_hash=sig_hash,
            signature_name=self._name,
            predictors_src=predictors_src,
            forward_src=forward_src,
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
