from __future__ import annotations

import ast
import inspect
import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable

import dspy
from dspy.primitives.code_interpreter import CodeInterpreterError

logger = logging.getLogger(__name__)

# Tool names the shim uses to call the host; must match the names registered in BridgeRuntime.
CONSTRUCT_TOOL = "__dspy_construct__"
CALL_TOOL = "__dspy_call__"

# Predictors the sandbox shim may construct and the host builds
BRIDGEABLE_KINDS = ("Predict", "ChainOfThought", "RLM", "CodeAct", "ProgramOfThought", "ReAct", "ReActV2")
# The shim's dspy.Signature(...) emits this marker so the host can rebuild a Signature.
SIGNATURE_MARKER = "__dspy_sig__"
# The shim passes tools by name (callables can't cross the JSON boundary); the host resolves the name
# back to the real tool object passed to dspy.Flex(tools=...).
TOOL_MARKER = "__dspy_tool__"

# Variable/identifier names used in the per-forward driver code (namespaced to avoid clashing with
# whatever the optimizer-authored module uses).
_INPUTS_VAR = "__dspy_flex_inputs"
_INSTANCE_VAR = "__dspy_flex_instance"
_OUT_VAR = "__dspy_flex_out"
_JSON_VAR = "__dspy_flex_json"


# The sandbox-side dspy shim, injected as text once per session.
SHIM_SETUP = (Path(__file__).parent / "_sandbox_shim.py").read_text(encoding="utf-8")


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


def _accepts_interpreter_factory(cls: type) -> bool:
    """True if ``cls.__init__`` takes an ``interpreter_factory`` parameter."""
    try:
        return "interpreter_factory" in inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def _resolve_signature(signature: Any) -> Any:
    """Turn a shim signature payload back into something a host predictor accepts."""
    if isinstance(signature, dict) and signature.get(SIGNATURE_MARKER):
        from dspy.signatures.signature import ensure_signature

        # marker always carries a string signature; ensure_signature applies instructions if given
        return ensure_signature(signature["signature"], signature.get("instructions"))
    return signature


def _signature_key(signature: Any) -> str:
    """A hashable, stable key for a signature payload (for construction idempotency)."""
    if isinstance(signature, str):
        return signature
    return json.dumps(signature, sort_keys=True, default=str)


def _jsonable(value: Any) -> Any:
    """Best-effort coercion of a predictor output field to a JSON-serializable value."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return value


def prediction_to_fields(pred: Any) -> dict[str, Any]:
    """Serialise a host ``dspy.Prediction`` (or dict) to a JSON-able field dict for the sandbox."""
    store = getattr(pred, "_store", None)
    if store is None:
        if isinstance(pred, dict):
            store = pred
        else:
            raise CodeInterpreterError(
                f"A bridged predictor must return a dspy.Prediction; got {type(pred).__name__}"
            )
    fields = {k: _jsonable(v) for k, v in dict(store).items()}
    try:
        json.dumps(fields)
    except TypeError as e:
        raise CodeInterpreterError(
            "A bridged predictor returned a field that cannot cross the sandbox boundary "
            f"(must be JSON-serializable): {e}"
        ) from e
    return fields


class _Session:
    """One sandbox interpreter pinned to one thread, plus its initialization bookkeeping."""

    __slots__ = ("calls", "constructed", "generation", "interp")

    def __init__(self, interp: Any) -> None:
        self.interp = interp
        self.generation = -1
        self.constructed = False
        self.calls = 0  # bridged predictor calls in the current forward (budget enforcement)


class BridgeRuntime:
    """Owns the sandbox session(s) and host-side bridge callbacks for one ``Flex``.

    Predictors are constructed on the host and attached to the ``Flex`` (so they stay optimizable and
    serializable). Sessions are thread-local, so parallel rollouts each get their own interpreter;
    construction is idempotent across sessions, so the host predictors are built once and shared.
    ``_max_predictor_calls`` caps bridged LM calls per ``forward``.

    User ``tools`` are registered with the interpreter (callable by name from sandbox) and
    resolved by name when passed to a bridged sub-predictor; tools authored inside the generated module
    live in the sandbox. A code-executing sub-predictor (RLM/CodeAct/ProgramOfThought)
    gets a fresh interpreter from the same factory, so its inner code runs in the backend chosen for
    Flex — except a shared bare instance, which it can't reuse (that would shut the session down after
    ``forward``), so there it falls back to its own default sandbox. See ``_sub_interpreter_factory``.
    """

    def __init__(self, flex: Any, factory: Callable[[], Any], max_predictor_calls: int | None = 100) -> None:
        self._flex = flex
        self._factory = factory
        self._max_predictor_calls = max_predictor_calls
        self._local = threading.local()
        self._all_interps: list[Any] = []
        self._lock = threading.Lock()
        self._registry: dict[str, str] = {}  # attr_name -> signature key (host predictors built once)
        self._generation = 0
        self._module_src: str | None = None
        self._class_name: str | None = None
        self._closed = False

    # -- registration -------------------------------------------------------

    def bridge_tools(self) -> dict[str, Callable[..., Any]]:
        return {CONSTRUCT_TOOL: self._construct, CALL_TOOL: self._call}

    def bind(self, module_src: str) -> None:
        """Record new source and invalidate existing sessions (they re-init lazily)."""
        self._module_src = module_src
        self._class_name = parse_module_class_name(module_src)
        self._generation += 1
        with self._lock:
            self._registry.clear()

    # -- sessions -----------------------------------------------------------

    def _get_session(self) -> _Session:
        if self._closed:
            raise CodeInterpreterError("dspy.Flex interpreter has been closed")
        sess: _Session | None = getattr(self._local, "sess", None)
        if sess is None:
            interp = self._factory()
            interp.tools.update(self.bridge_tools())
            interp.tools.update(self._tool_callables())  # user tools callable by name in the sandbox
            with self._lock:
                self._all_interps.append(interp)
            sess = _Session(interp)
            self._local.sess = sess
        if sess.generation != self._generation:
            sess.interp.execute(SHIM_SETUP)
            sess.interp.execute(self._module_src)  # defines the class in the sandbox
            sess.generation = self._generation
            sess.constructed = False
        return sess

    def _ensure_constructed(self, sess: _Session) -> None:
        if not sess.constructed:
            # Instantiating runs __init__ in the sandbox, which bridges predictor construction back to
            # the host (attaching them to the Flex instance). Idempotent on the host side.
            sess.interp.execute(f"{_INSTANCE_VAR}_probe = {self._class_name}()")
            sess.constructed = True

    def ensure_initialized(self) -> None:
        """Build the host predictors now (used at bind time so load_state/named_predictors see them)."""
        self._ensure_constructed(self._get_session())

    def forward(self, inputs: dict[str, Any]) -> Any:
        sess = self._get_session()
        self._ensure_constructed(sess)
        sess.calls = 0  # reset the per-forward predictor-call budget
        code = (
            f"{_INSTANCE_VAR} = {self._class_name}()\n"
            f"{_OUT_VAR} = {_INSTANCE_VAR}.forward(**{_INPUTS_VAR})\n"
            f"import json as {_JSON_VAR}\n"
            f"{_JSON_VAR}.dumps({_OUT_VAR}._fields if hasattr({_OUT_VAR}, '_fields') else {_OUT_VAR})"
        )
        result = sess.interp.execute(code, variables={_INPUTS_VAR: dict(inputs)})
        if not isinstance(result, str) or not result:
            raise CodeInterpreterError(
                "Sandboxed forward returned no serializable result; the generated forward must return "
                f"a dspy.Prediction (got {result!r})"
            )
        return dspy.Prediction(**json.loads(result))

    def shutdown(self) -> None:
        self._closed = True
        with self._lock:
            interps, self._all_interps = self._all_interps, []
        for interp in interps:
            try:
                interp.shutdown()
            except Exception:
                logger.warning("dspy.Flex: interpreter.shutdown() raised during close", exc_info=True)

    # -- host-side bridge callbacks --------------

    def _tool_callables(self) -> dict[str, Callable[..., Any]]:
        """User tools as ``name -> underlying callable`` to register so sandbox code can call them."""
        ctx = self._flex._flex_ctx.context_names()
        return {name: getattr(tool, "func", tool) for name, tool in ctx.items()}

    def _resolve_tool(self, name: str) -> Any:
        ctx = self._flex._flex_ctx.context_names()
        if name not in ctx:
            raise CodeInterpreterError(
                f"Sandboxed code referenced tool {name!r}, which was not passed to dspy.Flex(tools=...). "
                "Tools authored inside the generated module cannot be handed to a bridged sub-predictor."
            )
        return ctx[name]

    def _decode_tools(self, value: Any) -> Any:
        """Turn the shim's tool name-markers back into the real host tool objects."""
        if isinstance(value, dict) and TOOL_MARKER in value:
            return self._resolve_tool(value[TOOL_MARKER])
        if isinstance(value, list):
            return [self._decode_tools(v) for v in value]
        if isinstance(value, dict):
            return {k: self._decode_tools(v) for k, v in value.items()}
        return value

    def _sub_interpreter_factory(self) -> Any:
        """The Flex interpreter factory, handed to a bridged code-executing sub-predictor so its inner
        code runs in the same backend chosen for Flex. The sub-predictor creates and tears down a fresh
        interpreter per ``forward`` from this factory, isolating each rollout."""
        if getattr(self._flex, "_interpreter_shared", False):
            return None
        return self._factory

    def _build_predictor(self, kind: str, signature: Any, kwargs: dict[str, Any] | None) -> Any:
        cls = getattr(dspy, kind)
        extra = {k: self._decode_tools(v) for k, v in (kwargs or {}).items()}
        # A code-executing sub-predictor should run its inner code in the backend chosen for Flex, so
        # hand it the Flex interpreter factory (it makes and tears down a fresh interpreter per forward).
        # The sandbox code can't set this itself, since a live interpreter can't cross the boundary.
        if "interpreter_factory" not in extra and _accepts_interpreter_factory(cls):
            factory = self._sub_interpreter_factory()
            if factory is not None:
                extra["interpreter_factory"] = factory
        return cls(_resolve_signature(signature), **extra)

    def _construct(self, kind: str, signature: Any, attr_name: str, kwargs: dict[str, Any] | None = None) -> str:
        if kind not in BRIDGEABLE_KINDS:
            raise CodeInterpreterError(
                f"dspy.{kind} is not supported inside a sandboxed dspy.Flex yet "
                f"(bridgeable: {', '.join(BRIDGEABLE_KINDS)})"
            )
        sig_key = _signature_key(signature)
        with self._lock:
            if self._registry.get(attr_name) == sig_key and getattr(self._flex, attr_name, None) is not None:
                return attr_name  # already built with the same signature; reuse (preserves demos)
            predictor = self._build_predictor(kind, signature, kwargs)
            # Attach under attr_name so it keeps a canonical name in named_parameters()/state; the
            # handle IS that name, so _call resolves it with getattr.
            setattr(self._flex, attr_name, predictor)
            if attr_name not in self._flex._attached_names:
                self._flex._attached_names.append(attr_name)
            self._registry[attr_name] = sig_key
        return attr_name

    def _call(self, handle: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        # Cap calls: generated glue could loop over a predictor and run up the host LM bill.
        sess: _Session | None = getattr(self._local, "sess", None)
        if sess is not None and self._max_predictor_calls is not None:
            sess.calls += 1
            if sess.calls > self._max_predictor_calls:
                raise CodeInterpreterError(
                    f"Sandboxed dspy.Flex forward exceeded its predictor-call budget "
                    f"({self._max_predictor_calls}). Raise max_predictor_calls if this is expected."
                )
        predictor = getattr(self._flex, handle, None)
        if predictor is None:
            raise CodeInterpreterError(f"Unknown predictor handle: {handle!r}")
        out = predictor(**(inputs or {}))
        return prediction_to_fields(out)
