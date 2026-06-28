"""dspy.Flex(interpreter=...) runs generated code inside a sandbox via the run-in-interpreter bridge.

Two layers of coverage:

* Unit (no Deno): the interpreter is normalized to a factory; the host-side bridge callbacks
  (``_construct``/``_call``) build real predictors on the host, attach them under their canonical
  names, are idempotent, and run via the configured LM. These exercise everything the sandbox would
  drive, without needing Deno.
* Integration (Deno required, skipped otherwise): an optimized CoT-style module — with optimizer
  authored glue like ``import re`` — actually runs inside a ``dspy.PythonInterpreter`` with the LM call
  bridged back to the host, and adversarial file access is contained to the sandbox.
"""

from __future__ import annotations

import json
import shutil
import textwrap
from unittest.mock import patch as mock_patch

import pytest

import dspy
from dspy.flex import Flex, bridge
from dspy.primitives.code_interpreter import CodeInterpreterError
from dspy.utils.dummies import DummyLM
from tests.mock_interpreter import MockInterpreter

deno_required = pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed")


class Doubler(dspy.Signature):
    """Return double the input value."""

    value: int = dspy.InputField()
    result: int = dspy.OutputField()


def shout(text: str) -> str:
    """Uppercase helper used to exercise user tools through the bridge."""
    return text.upper()


class ShoutSig(dspy.Signature):
    """Uppercase the input text."""

    text: str = dspy.InputField()
    out: str = dspy.OutputField()


# =============================================================================
# Normalization / construction (no Deno)
# =============================================================================


def test_no_interpreter_keeps_in_process_path() -> None:
    flex = Flex(Doubler)
    assert flex._bridge is None
    assert flex._interpreter_factory is None
    # Unchanged baseline behavior.
    assert flex.module_src and "dspy.RLM(" in flex.module_src


def test_instance_is_wrapped_as_factory_with_warning() -> None:
    mock = MockInterpreter()
    with mock_patch("dspy.flex.flex.logger") as log:
        flex = Flex(Doubler, interpreter=mock)
    assert flex._bridge is not None
    assert flex._interpreter_factory is not None
    # The factory yields the very same shared instance, and the user was warned about sharing.
    assert flex._interpreter_factory() is mock
    assert log.warning.called
    assert "shared across all forward" in log.warning.call_args[0][0]


def test_factory_is_used_as_is_without_warning() -> None:
    with mock_patch("dspy.flex.flex.logger") as log:
        flex = Flex(Doubler, interpreter=lambda: MockInterpreter())
    assert flex._bridge is not None
    assert not log.warning.called


def test_bad_interpreter_type_raises() -> None:
    with pytest.raises(TypeError):
        Flex(Doubler, interpreter=123)  # type: ignore[arg-type]


def test_tools_plus_interpreter_is_supported() -> None:
    flex = Flex(Doubler, tools=[shout], interpreter=lambda: MockInterpreter())
    assert flex._bridge is not None
    # the user tool is registered (name -> callable) so sandbox glue can call it by name
    assert flex._bridge._tool_callables()["shout"] is shout


# =============================================================================
# Host bridge callbacks (no Deno) — simulate what the sandbox drives
# =============================================================================


def _bridged_flex():
    # MockInterpreter doesn't execute Python, so eager init attaches nothing; we drive the host
    # bridge callbacks directly, exactly as the sandbox shim would.
    return Flex(Doubler, interpreter=lambda: MockInterpreter())


def _bridged_flex_with(**kwargs):
    return Flex(Doubler, interpreter=lambda: MockInterpreter(), **kwargs)


def test_construct_attaches_real_predictor_under_canonical_name() -> None:
    flex = _bridged_flex()
    handle = flex._bridge._construct("ChainOfThought", "value: int -> result: int", "solve", {})
    assert handle == "solve"
    assert isinstance(flex.solve, dspy.ChainOfThought)
    # Keeps its canonical predictor name (not "_bridge_predictors[...]") so state round-trips.
    names = [n for n, _ in flex.named_predictors()]
    assert "solve.predict" in names
    assert "solve" in flex._attached_names


def test_construct_is_idempotent_but_rebuilds_on_signature_change() -> None:
    flex = _bridged_flex()
    flex._bridge._construct("ChainOfThought", "value: int -> result: int", "solve", {})
    first = flex.solve
    # Same (kind, signature) -> reuse the same object (preserves any optimized demos/instructions).
    flex._bridge._construct("ChainOfThought", "value: int -> result: int", "solve", {})
    assert flex.solve is first
    # Changed signature -> rebuild.
    flex._bridge._construct("ChainOfThought", "value: int -> result: str", "solve", {})
    assert flex.solve is not first


def test_construct_rejects_non_bridgeable_kind() -> None:
    flex = _bridged_flex()
    with pytest.raises(CodeInterpreterError):
        flex._bridge._construct("Retrieve", "value -> result", "r", {})


def test_call_runs_predictor_via_host_lm() -> None:
    flex = _bridged_flex()
    flex._bridge._construct("ChainOfThought", "value: int -> result: int", "solve", {})
    dspy.configure(lm=DummyLM([{"reasoning": "two doubled is four", "result": "4"}]))
    fields = flex._bridge._call("solve", {"value": 2})
    assert fields["result"] == 4
    assert "reasoning" in fields


def test_call_unknown_handle_raises() -> None:
    flex = _bridged_flex()
    with pytest.raises(CodeInterpreterError):
        flex._bridge._call("nope", {"value": 1})


def test_tool_name_markers_resolve_to_real_tools() -> None:
    # The shim passes tools to a bridged sub-predictor by name; the host resolves them back.
    flex = Flex(Doubler, tools=[shout], interpreter=lambda: MockInterpreter())
    assert flex._bridge._decode_tools([{"__dspy_tool__": "shout"}]) == [shout]
    with pytest.raises(CodeInterpreterError):
        flex._bridge._decode_tools({"__dspy_tool__": "not_a_tool"})  # module-authored / unknown tool


def test_interpreter_is_not_serialized() -> None:
    flex = _bridged_flex()
    flex._bridge._construct("ChainOfThought", "value: int -> result: int", "solve", {})
    state = flex.dump_state()
    assert state.get("module_src")  # the code persists
    assert "interpreter" not in json.dumps(state)  # the runtime sandbox does not


# =============================================================================
# bridge.py helpers (no Deno)
# =============================================================================


def test_parse_module_class_name_prefers_forward() -> None:
    src = textwrap.dedent(
        """
        class Helper:
            pass

        class MyModule(dspy.Module):
            def forward(self, **inputs):
                return dspy.Prediction()
        """
    )
    assert bridge.parse_module_class_name(src) == "MyModule"


def test_prediction_to_fields_rejects_non_jsonable() -> None:
    assert bridge.prediction_to_fields(dspy.Prediction(a=1, b="x")) == {"a": 1, "b": "x"}
    with pytest.raises(CodeInterpreterError):
        bridge.prediction_to_fields(dspy.Prediction(a=object()))


# =============================================================================
# Backend-agnostic shim + predictor-call budget (Phase 2, no Deno)
# =============================================================================


def test_shim_reaches_host_only_via_registered_tools() -> None:
    # The bridge must not depend on Deno/Pyodide internals — only on the CodeInterpreter.tools
    # contract — so any backend (a local microVM, gVisor, ...) can drive it.
    shim = bridge.SHIM_SETUP
    assert "pyodide" not in shim.lower()
    assert "_js_tool_call" not in shim
    assert "run_sync" not in shim
    # It does call the registered host tools by name.
    assert bridge.CONSTRUCT_TOOL in shim
    assert bridge.CALL_TOOL in shim


def test_shim_file_literals_match_bridge_constants() -> None:
    # The shim lives in a sibling .py file with literal protocol strings (no interpolation); guard
    # against drift between those literals and bridge.py's host-side constants.
    shim = bridge.SHIM_SETUP
    for token in (bridge.CONSTRUCT_TOOL, bridge.CALL_TOOL, bridge.TOOL_MARKER, bridge.SIGNATURE_MARKER):
        assert token in shim, f"{token!r} missing from the shim file"
    for kind in bridge.BRIDGEABLE_KINDS:
        assert kind in shim, f"bridgeable kind {kind!r} missing from the shim file"


def test_max_predictor_calls_is_plumbed_to_bridge() -> None:
    flex = _bridged_flex_with(max_predictor_calls=7)
    assert flex._bridge._max_predictor_calls == 7


def test_predictor_call_budget_is_enforced() -> None:
    flex = _bridged_flex_with(max_predictor_calls=2)
    flex._bridge._construct("ChainOfThought", "value: int -> result: int", "solve", {})
    dspy.configure(lm=DummyLM([{"reasoning": "r", "result": "1"}, {"reasoning": "r", "result": "2"}]))
    flex._bridge._get_session()  # a session must exist for per-forward budget tracking
    flex._bridge._local.sess.calls = 0
    flex._bridge._call("solve", {"value": 1})  # 1st
    flex._bridge._call("solve", {"value": 1})  # 2nd
    with pytest.raises(CodeInterpreterError, match="budget"):
        flex._bridge._call("solve", {"value": 1})  # 3rd exceeds the cap


def test_predictor_call_budget_can_be_disabled() -> None:
    flex = _bridged_flex_with(max_predictor_calls=None)
    assert flex._bridge._max_predictor_calls is None
    flex._bridge._construct("ChainOfThought", "value: int -> result: int", "solve", {})
    dspy.configure(lm=DummyLM([{"reasoning": "r", "result": str(i)} for i in range(5)]))
    flex._bridge._get_session()
    flex._bridge._local.sess.calls = 0
    for _ in range(5):  # no cap -> all calls go through
        flex._bridge._call("solve", {"value": 1})


# =============================================================================
# Integration (Deno) — the real run-in-interpreter bridge
# =============================================================================

# An optimized CoT module with optimizer-authored glue (mirrors tests/flex_mathword_optimized.json):
# the ChainOfThought call must bridge to the host; the import/regex/int parsing run in the sandbox.
COT_GLUE_MODULE = textwrap.dedent(
    """
    class DoublerModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.solve = dspy.ChainOfThought("value: int -> result: str")

        def forward(self, **inputs):
            r = self.solve(value=inputs["value"])
            try:
                answer = int(r.result)
            except (ValueError, TypeError):
                import re
                m = re.search(r"-?\\d+", str(r.result))
                answer = int(m.group()) if m else 0
            return dspy.Prediction(result=answer)
    """
).strip()


@deno_required
def test_optimized_cot_runs_in_sandbox_with_lm_bridged() -> None:
    dspy.configure(lm=DummyLM([{"reasoning": "double of 21 is 42", "result": "the result is 42"}]))
    flex = Flex(Doubler, interpreter=lambda: dspy.PythonInterpreter())
    try:
        flex._bind_code(COT_GLUE_MODULE)
        # forward() runs entirely in the sandbox; self.solve(...) bridges to the host DummyLM, and the
        # `import re` / int-parsing glue executes inside Pyodide.
        out = flex(value=21)
        assert isinstance(out, dspy.Prediction)
        assert out.result == 42
        # It really took the bridge path (no in-process forward bound), and the predictor is a real
        # host object discoverable for optimization/serialization.
        assert flex._forward_impl is None
        assert isinstance(flex.solve, dspy.ChainOfThought)
    finally:
        flex.close()


# Adversarial module: tries to read a host file. In the sandbox this hits Pyodide's virtual FS, which
# has no such file, so it is contained — the host file is never exposed.
def _evil_read_module(host_path: str) -> str:
    return textwrap.dedent(
        f"""
        class DoublerModule(dspy.Module):
            def __init__(self):
                super().__init__()

            def forward(self, **inputs):
                try:
                    with open({host_path!r}) as f:
                        leaked = f.read()
                except Exception as e:
                    leaked = "BLOCKED:" + type(e).__name__
                return dspy.Prediction(result=leaked)
        """
    ).strip()


@deno_required
def test_bridged_program_survives_save_load(tmp_path) -> None:
    dspy.configure(lm=DummyLM([{"reasoning": "double of 21 is 42", "result": "the result is 42"}]))
    program = Flex(Doubler, interpreter=lambda: dspy.PythonInterpreter())
    path = tmp_path / "prog.json"
    try:
        program._bind_code(COT_GLUE_MODULE)
        program.save(path)
    finally:
        program.close()

    # The interpreter is a runtime dependency (like the LM): the saved code persists, the sandbox does not.
    assert "interpreter" not in path.read_text()

    # Reconstruct with a fresh interpreter factory and load the saved code.
    reloaded = Flex(Doubler, interpreter=lambda: dspy.PythonInterpreter())
    try:
        reloaded.load(path)
        assert reloaded.module_src and "self.solve" in reloaded.module_src
        assert isinstance(reloaded.solve, dspy.ChainOfThought)
        out = reloaded(value=21)  # still runs in the sandbox, LM bridged to the host
        assert out.result == 42
    finally:
        reloaded.close()


# A module whose glue calls a user-provided tool directly (no LM): proves Flex tools are callable
# from inside the sandbox via the host bridge.
TOOL_GLUE_MODULE = textwrap.dedent(
    """
    class ShoutModule(dspy.Module):
        def __init__(self):
            super().__init__()

        def forward(self, **inputs):
            return dspy.Prediction(out=shout(inputs["text"]))
    """
).strip()


@deno_required
def test_user_tool_is_callable_from_sandbox_glue() -> None:
    flex = Flex(ShoutSig, tools=[shout], interpreter=lambda: dspy.PythonInterpreter())
    try:
        flex._bind_code(TOOL_GLUE_MODULE)
        # shout() runs on the host (bridged from the sandbox); no LM involved.
        assert flex(text="hello").out == "HELLO"
    finally:
        flex.close()


RUNAWAY_MODULE = textwrap.dedent(
    """
    class DoublerModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.solve = dspy.ChainOfThought("value: int -> result: str")

        def forward(self, **inputs):
            n = 0
            for _ in range(100):           # runaway glue: would drive 100 host LM calls
                self.solve(value=inputs["value"])
                n += 1
            return dspy.Prediction(result=n)
    """
).strip()


@deno_required
def test_runaway_glue_is_stopped_by_budget_in_sandbox() -> None:
    dspy.configure(lm=DummyLM([{"reasoning": "r", "result": "1"} for _ in range(50)]))
    flex = Flex(Doubler, interpreter=lambda: dspy.PythonInterpreter(), max_predictor_calls=3)
    try:
        flex._bind_code(RUNAWAY_MODULE)
        with pytest.raises(Exception) as excinfo:  # the budget error propagates out of the sandbox
            flex(value=1)
        assert "budget" in str(excinfo.value).lower()
    finally:
        flex.close()


@deno_required
def test_adversarial_host_file_access_is_contained(tmp_path) -> None:
    secret = tmp_path / "host_secret.txt"
    secret.write_text("TOP-SECRET")
    assert secret.read_text() == "TOP-SECRET"  # readable in-process on the host

    class Leak(dspy.Signature):
        value: int = dspy.InputField()
        result: str = dspy.OutputField()

    flex = Flex(Leak, interpreter=lambda: dspy.PythonInterpreter())
    try:
        flex._bind_code(_evil_read_module(str(secret)))
        out = flex(value=1)
        # The sandbox could not reach the host file; the secret never crossed the boundary.
        assert "TOP-SECRET" not in str(out.result)
        assert str(out.result).startswith("BLOCKED")
    finally:
        flex.close()
