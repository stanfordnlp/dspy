from __future__ import annotations

import hashlib
import json
import logging
import types
from pathlib import Path
from typing import Any

import dspy
from dspy.flex.codegen import FlexContext, generate, repair
from dspy.flex.exploration import ExplorationStore, FlexEvent, candidate_id
from dspy.flex.manifest import ManifestStore
from dspy.flex.persistence import PersistedFlex, parse_persisted_file, render_persisted_file
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
    """A DSPy Module whose ``forward()`` body is authored by an LM from a Signature.

    Args:
        signature: A ``dspy.Signature`` class (or string) declaring inputs/outputs.
        persist_to: Path to a checked-in ``.py`` file holding the generated code.
            When set, the file is loaded if its signature hash matches; otherwise
            it is (re-)generated and written. When None, the implementation lives
            only in memory.
        context: Optional list of ``dspy.Tool`` / callables / style note strings to
            inject into codegen and into the runtime exec namespace.
        codegen_lm: LM to use for generation. Defaults to ``dspy.settings.lm``.
        flex_id: Stable identifier in the manifest. Defaults to the Signature class
            name (or a hash of the signature when ``persist_to`` is None).
        auto_repair: When True (default), Flex tries to recover from broken
            implementations by re-invoking the codegen LM with the broken body
            and the exception text. Runs at most once at bind time and at most
            once at runtime per process. Set False to surface errors directly.

    Persistence layout: when ``persist_to`` is set, the generated source goes
    to that file and bookkeeping goes to ``<persist_to>.parent/.flex/`` —
    ``manifest.json`` (shared across all Flexes in that directory) and per-flex
    history under ``.flex/<flex_id>/``. When ``persist_to=None`` the
    implementation lives only in memory and no bookkeeping is written.
    """

    def __init__(
        self,
        signature: Any,
        *,
        persist_to: str | Path | None = None,
        context: list[Any] | None = None,
        codegen_lm: dspy.LM | None = None,
        flex_id: str | None = None,
        auto_repair: bool = True,
    ):
        super().__init__()

        from dspy.signatures.signature import ensure_signature

        self._signature_cls = ensure_signature(signature)
        self._persist_to: Path | None = Path(persist_to) if persist_to else None
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

        self._forward_src: str | None = None
        self._predictors_src: str | None = None
        self._predictor_names: list[str] = []
        self._forward_impl: Any = None
        self._auto_repair = auto_repair
        self._runtime_repair_used = False

        sig_hash = self._signature_hash()
        self._flex_id = flex_id or getattr(self._signature_cls, "__name__", None) or f"flex_{sig_hash[:8]}"

        # `.flex/` always lives next to `persist_to`. In-memory mode (no
        # persist_to) → root=None → the store no-ops on every write.
        flex_root = self._persist_to.parent if self._persist_to is not None else None
        self._flex_root: Path | None = flex_root
        self._exploration = ExplorationStore(flex_root, self._flex_id)

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
        state["_flex"] = {
            "forward_src": self._forward_src,
            "predictors_src": self._predictors_src,
            "signature_hash": self._signature_hash(),
            "flex_id": self._flex_id,
        }
        return state

    def load_state(self, state: dict[str, Any], *, allow_unsafe_lm_state: bool = False) -> None:
        flex_state = state.pop("_flex", None) if isinstance(state, dict) else None
        if flex_state and flex_state.get("forward_src") and flex_state.get("predictors_src"):
            self._bind_code(flex_state["predictors_src"], flex_state["forward_src"])
        if state:
            super().load_state(state, allow_unsafe_lm_state=allow_unsafe_lm_state)

    def _lazy_implement(self, *, force: bool = False) -> None:
        if self._forward_src and self._predictors_src and not force:
            return

        sig_hash = self._signature_hash()

        # Seed for regeneration: the current on-disk body, carried into codegen
        # when the signature changed so hand edits aren't thrown away.
        seed: tuple[str, str] | None = None

        if self._persist_to and self._persist_to.exists() and not force:
            stored = self._read_persisted()
            if stored is not None:
                body_id = candidate_id(stored.predictors_src, stored.forward_src)
                body_edited = stored.body_hash is not None and body_id != stored.body_hash

                if stored.signature_hash == sig_hash:
                    # Signature unchanged — run exactly what's on disk, edited or not.
                    try:
                        self._bind_code(stored.predictors_src, stored.forward_src)
                    except Exception as err:
                        if not self._auto_repair:
                            raise
                        broken = (stored.predictors_src, stored.forward_src)
                        fixed_predictors, fixed_forward = self._repair_after_bind_failure(
                            broken=broken,
                            error=err,
                            signature_hash=sig_hash,
                            parents=self._latest_manifest_parents(),
                        )
                        self._accept_fixed(
                            fixed_predictors,
                            fixed_forward,
                            sig_hash=sig_hash,
                            parents=[candidate_id(*broken)],
                            notes=f"auto-repair after load bind error: {type(err).__name__}",
                        )
                        return
                    if body_edited:
                        self._honor_manual_edit(stored, sig_hash, body_id)
                    else:
                        self._exploration.record(
                            FlexEvent.LOAD,
                            predictors_src=stored.predictors_src,
                            forward_src=stored.forward_src,
                            signature_hash=sig_hash,
                            extra={"source_path": str(self._persist_to)},
                        )
                        logger.info(
                            "dspy.Flex %r: loaded existing implementation from %s",
                            self._flex_id,
                            self._persist_to,
                        )
                        # Legacy file written before body hashes existed: backfill
                        # one so future edits are detectable.
                        if stored.body_hash is None:
                            self._write_persisted(stored.predictors_src, stored.forward_src, sig_hash)
                    return

                # Signature changed — regenerate, seeded from the current body.
                seed = (stored.predictors_src, stored.forward_src)
                if body_edited:
                    # Preserve the hand edit in the ledger so the codegen event's
                    # parent (the seed) refers to a real candidate.
                    self._exploration.record(
                        FlexEvent.MANUAL_EDIT,
                        predictors_src=stored.predictors_src,
                        forward_src=stored.forward_src,
                        signature_hash=stored.signature_hash,
                        extra={
                            "source_path": str(self._persist_to),
                            "reason": "manual edit seeding regeneration",
                        },
                    )
                logger.info(
                    "dspy.Flex %r: signature hash mismatch on %s — regenerating from %s implementation.",
                    self._flex_id,
                    self._persist_to,
                    "hand-edited" if body_edited else "previous",
                )

        predictors_src, forward_src = generate(self._flex_ctx, lm=self._codegen_lm, seed=seed)
        seed_parents = [candidate_id(*seed)] if seed is not None else None
        codegen_parents = seed_parents
        notes = "regenerated from seed (signature changed)" if seed is not None else "initial codegen"
        try:
            self._bind_code(predictors_src, forward_src)
        except Exception as err:
            if not self._auto_repair:
                raise
            broken = (predictors_src, forward_src)
            predictors_src, forward_src = self._repair_after_bind_failure(
                broken=broken,
                error=err,
                signature_hash=sig_hash,
                parents=seed_parents,
            )
            codegen_parents = [candidate_id(*broken)]
            notes = f"auto-repair after codegen bind error: {type(err).__name__}"
        cid = self._exploration.record(
            FlexEvent.CODEGEN,
            predictors_src=predictors_src,
            forward_src=forward_src,
            signature_hash=sig_hash,
            parents=codegen_parents,
        )

        if self._persist_to is not None:
            self._write_persisted(predictors_src, forward_src, sig_hash)
            assert self._flex_root is not None
            manifest = ManifestStore(self._flex_root)
            version_id = manifest.append_version(
                flex_id=self._flex_id,
                src_path=self._persist_to,
                signature_hash=sig_hash,
                candidate_id=cid,
                parents=codegen_parents,
                notes=notes,
            )
            self._exploration.record(
                FlexEvent.ACCEPT,
                predictors_src=predictors_src,
                forward_src=forward_src,
                signature_hash=sig_hash,
                extra={"version_id": version_id, "src_path": str(self._persist_to)},
            )
            logger.info("dspy.Flex %r: wrote new implementation to %s", self._flex_id, self._persist_to)
        else:
            logger.warning(
                "dspy.Flex %r: persist_to=None — implementation is in-memory only and will be "
                "regenerated next process start. Set persist_to= for production.",
                self._flex_id,
            )

    def _honor_manual_edit(self, stored: PersistedFlex, sig_hash: str, body_id: str) -> None:
        """Accept a hand-edited file: record it, refresh the body hash, and version it.

        Called when the signature is unchanged but the body hash no longer matches
        the persisted code — i.e. the user edited the file by hand. The edit is the
        running implementation (already bound by the caller). We log a
        ``manual_edit`` event, rewrite the file so its body hash
        matches again, and append a manifest version.
        """
        self._exploration.record(
            FlexEvent.MANUAL_EDIT,
            predictors_src=stored.predictors_src,
            forward_src=stored.forward_src,
            signature_hash=sig_hash,
            extra={"source_path": str(self._persist_to)},
        )
        # Refresh the body hash so subsequent runs see a pristine file.
        self._write_persisted(stored.predictors_src, stored.forward_src, sig_hash)

        if self._flex_root is not None:
            manifest = ManifestStore(self._flex_root)
            prev = manifest.latest(self._flex_id)
            parents = [prev["candidate_id"]] if prev and prev.get("candidate_id") else []
            version_id = manifest.append_version(
                flex_id=self._flex_id,
                src_path=self._persist_to,
                signature_hash=sig_hash,
                candidate_id=body_id,
                parents=parents,
                notes="manual edit",
            )
            self._exploration.record(
                FlexEvent.ACCEPT,
                predictors_src=stored.predictors_src,
                forward_src=stored.forward_src,
                signature_hash=sig_hash,
                extra={"version_id": version_id, "src_path": str(self._persist_to)},
            )
        logger.info(
            "dspy.Flex %r: honored hand-edited implementation in %s (recorded a `manual_edit` "
            "event and a new manifest version).",
            self._flex_id,
            self._persist_to,
        )

    def _latest_manifest_parents(self) -> list[str] | None:
        """Candidate IDs of the last accepted manifest entry, for REPAIR parentage."""
        if self._flex_root is None:
            return None
        prev = ManifestStore(self._flex_root).latest(self._flex_id)
        if prev and prev.get("candidate_id"):
            return [prev["candidate_id"]]
        return None

    def _accept_fixed(
        self,
        predictors_src: str,
        forward_src: str,
        *,
        sig_hash: str,
        parents: list[str],
        notes: str,
    ) -> None:
        """Record a CODEGEN+ACCEPT pair, write the file, and append a manifest version.

        Used after auto-repair: the new (fixed) implementation is already bound.
        """
        cid = self._exploration.record(
            FlexEvent.CODEGEN,
            predictors_src=predictors_src,
            forward_src=forward_src,
            signature_hash=sig_hash,
            parents=parents,
        )
        if self._persist_to is None or self._flex_root is None:
            return
        self._write_persisted(predictors_src, forward_src, sig_hash)
        version_id = ManifestStore(self._flex_root).append_version(
            flex_id=self._flex_id,
            src_path=self._persist_to,
            signature_hash=sig_hash,
            candidate_id=cid,
            parents=parents,
            notes=notes,
        )
        self._exploration.record(
            FlexEvent.ACCEPT,
            predictors_src=predictors_src,
            forward_src=forward_src,
            signature_hash=sig_hash,
            extra={"version_id": version_id, "src_path": str(self._persist_to)},
        )
        logger.info(
            "dspy.Flex %r: wrote auto-repaired implementation to %s (manifest version %d).",
            self._flex_id,
            self._persist_to,
            version_id,
        )

    def _repair_after_bind_failure(
        self,
        *,
        broken: tuple[str, str],
        error: BaseException,
        signature_hash: str,
        parents: list[str] | None,
    ) -> tuple[str, str]:
        """Invoke the repair codegen LM after ``_bind_code`` raised.

        Records the broken candidate under a ``REPAIR`` event with the supplied
        ``parents``, asks the codegen LM to fix it, and binds the result. The
        bind of the *repaired* code is not wrapped — if the repair itself is
        broken, the user sees that as a hard error rather than an infinite
        repair loop. Returns the fixed ``(predictors_src, forward_src)``.

        The caller is responsible for the subsequent ``CODEGEN`` event, file
        write, and manifest append using the returned sources.
        """
        broken_predictors, broken_forward = broken
        error_text = f"{type(error).__name__}: {error}"
        logger.warning(
            "dspy.Flex %r: bind failure (%s) — running auto-repair. "
            "Set auto_repair=False to surface the error directly.",
            self._flex_id,
            error_text,
        )
        self._exploration.record(
            FlexEvent.REPAIR,
            predictors_src=broken_predictors,
            forward_src=broken_forward,
            signature_hash=signature_hash,
            parents=parents,
            extra={"failure_kind": "bind", "error": error_text},
        )
        fixed_predictors, fixed_forward = repair(
            self._flex_ctx,
            broken=broken,
            failure_kind="bind",
            error_text=error_text,
            lm=self._codegen_lm,
        )
        self._bind_code(fixed_predictors, fixed_forward)
        return fixed_predictors, fixed_forward

    def _bind_code(self, predictors_src: str, forward_src: str) -> None:
        """Exec and attach both source artifacts to ``self``."""
        ctx_names = self._flex_ctx.context_names()
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
        # Bind to `_forward_impl` instead of `forward`. The class-level
        # ``forward`` wraps this with runtime auto-repair.
        self._forward_impl = types.MethodType(forward_fn, self)

        self._predictors_src = predictors_src
        self._forward_src = forward_src

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the LM-authored ``_forward_impl`` with runtime auto-repair.

        If a call raises one of the narrow user-code errors in
        ``_RUNTIME_REPAIRABLE_ERRORS`` and ``auto_repair`` is on, Flex
        re-invokes the codegen LM with the broken body + the exception, swaps
        in the fix, and re-runs the call once. Subsequent failures (or
        non-repairable exceptions like a tool/LM error) propagate unchanged.
        """
        if self._forward_impl is None:
            raise RuntimeError(
                f"dspy.Flex {self._flex_id!r}: no implementation bound. "
                "Did __init__ or implement() fail?"
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
        """Regenerate, rebind, log, and persist after a runtime exception in forward().

        Mirrors the bind-time repair flow but is driven by an exception raised
        while ``_forward_impl`` was running. The broken candidate is the
        currently-bound code. Records ``REPAIR`` → ``CODEGEN`` → ``ACCEPT``,
        parented to the last accepted manifest entry. If repair codegen
        produces another bind-broken body, the bind error propagates — we do
        not loop on repair.
        """
        assert self._predictors_src is not None and self._forward_src is not None
        broken = (self._predictors_src, self._forward_src)
        sig_hash = self._signature_hash()
        error_text = f"{type(error).__name__}: {error}"
        logger.warning(
            "dspy.Flex %r: runtime failure (%s) — running auto-repair. "
            "Set auto_repair=False to surface the error directly.",
            self._flex_id,
            error_text,
        )
        self._exploration.record(
            FlexEvent.REPAIR,
            predictors_src=broken[0],
            forward_src=broken[1],
            signature_hash=sig_hash,
            parents=self._latest_manifest_parents(),
            extra={"failure_kind": "runtime", "error": error_text},
        )
        fixed_predictors, fixed_forward = repair(
            self._flex_ctx,
            broken=broken,
            failure_kind="runtime",
            error_text=error_text,
            lm=self._codegen_lm,
        )
        self._bind_code(fixed_predictors, fixed_forward)
        self._accept_fixed(
            fixed_predictors,
            fixed_forward,
            sig_hash=sig_hash,
            parents=[candidate_id(*broken)],
            notes=f"auto-repair after runtime error: {type(error).__name__}",
        )

    def _read_persisted(self) -> PersistedFlex | None:
        if self._persist_to is None or not self._persist_to.exists():
            return None
        parsed = parse_persisted_file(self._persist_to.read_text(encoding="utf-8"))
        if parsed is None:
            logger.warning(
                "dspy.Flex %r: persisted file at %s is missing expected markers",
                self._flex_id,
                self._persist_to,
            )
        return parsed

    def _write_persisted(self, predictors_src: str, forward_src: str, sig_hash: str) -> None:
        assert self._persist_to is not None
        self._persist_to.parent.mkdir(parents=True, exist_ok=True)
        body = render_persisted_file(
            signature_hash=sig_hash,
            body_hash=candidate_id(predictors_src, forward_src),
            flex_id=self._flex_id,
            signature_name=getattr(self._signature_cls, "__name__", "Signature"),
            predictors_src=predictors_src,
            forward_src=forward_src,
        )
        self._persist_to.write_text(body, encoding="utf-8")

    def _signature_hash(self) -> str:
        cls: Any = self._signature_cls
        canonical = {
            "name": getattr(cls, "__name__", ""),
            "instructions": getattr(cls, "instructions", "") or "",
            "input_fields": [
                (n, _annotation_repr(f.annotation)) for n, f in cls.input_fields.items()
            ],
            "output_fields": [
                (n, _annotation_repr(f.annotation)) for n, f in cls.output_fields.items()
            ],
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
