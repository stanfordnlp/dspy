"""dspy.Flex(interpreter_factory=...) runs generated code inside a sandbox via the run-in-interpreter bridge.

Two layers of coverage:

* Unit (no Deno): the interpreter_factory is validated; the host-side bridge callbacks
  (``_construct``/``_call``) build real predictors on the host, attach them under their canonical
  names, are idempotent, and run via the configured LM. These exercise everything the sandbox would
  drive, without needing Deno.
* Integration (Deno required, skipped otherwise): an optimized CoT-style module — with optimizer
  authored glue like ``import re`` — actually runs inside a ``dspy.PythonInterpreter`` with the LM call
  bridged back to the host, and adversarial file access is contained to the sandbox.
"""

from __future__ import annotations

import inspect
import json
import shutil
import textwrap

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


def test_default_interpreter_factory_is_python_interpreter() -> None:
    # Like dspy.RLM, interpreter_factory defaults to dspy.PythonInterpreter (a class used as a
    # zero-arg factory). Inspect the constructor default so this needs no Deno.
    default = inspect.signature(Flex.__init__).parameters["interpreter_factory"].default
    assert default is dspy.PythonInterpreter


@deno_required
def test_default_is_sandbox() -> None:
    # With no interpreter_factory argument, Flex builds a PythonInterpreter sandbox (needs Deno).
    flex = Flex(Doubler)
    try:
        assert flex._bridge is not None
        assert flex._interpreter_factory is dspy.PythonInterpreter
        assert isinstance(flex._interpreter_factory(), dspy.PythonInterpreter)
    finally:
        flex.close()


def test_bare_instance_is_rejected() -> None:
    # There is no in-process mode and no shared-instance mode: like dspy.RLM, the constructor takes a
    # factory, not a live interpreter. Passing an instance is a TypeError.
    with pytest.raises(TypeError):
        Flex(Doubler, interpreter_factory=MockInterpreter())  # type: ignore[arg-type]


def test_none_interpreter_factory_is_rejected() -> None:
    # None used to mean "run in-process"; that path is gone, so None is now invalid.
    with pytest.raises(TypeError):
        Flex(Doubler, interpreter_factory=None)  # type: ignore[arg-type]


def test_factory_is_used_as_is() -> None:
    flex = Flex(Doubler, interpreter_factory=lambda: MockInterpreter())
    assert flex._bridge is not None
    assert callable(flex._interpreter_factory)


def test_bad_interpreter_type_raises() -> None:
    with pytest.raises(TypeError):
        Flex(Doubler, interpreter_factory=123)  # type: ignore[arg-type]


def test_tools_plus_interpreter_is_supported() -> None:
    flex = Flex(Doubler, tools=[shout], interpreter_factory=lambda: MockInterpreter())
    assert flex._bridge is not None
    # the user tool is registered (name -> callable) so sandbox glue can call it by name
    assert flex._bridge._tool_callables()["shout"] is shout


# =============================================================================
# Host bridge callbacks (no Deno) — simulate what the sandbox drives
# =============================================================================


def _bridged_flex():
    # MockInterpreter doesn't execute Python, so eager init attaches nothing; we drive the host
    # bridge callbacks directly, exactly as the sandbox shim would.
    return Flex(Doubler, interpreter_factory=lambda: MockInterpreter())


def _bridged_flex_with(**kwargs):
    return Flex(Doubler, interpreter_factory=lambda: MockInterpreter(), **kwargs)


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


def test_react_kinds_are_bridgeable_and_construct_on_host() -> None:
    # ReAct/ReActV2 are tool-calling agents: they run NO inner code (so take no interpreter) and call
    # named tools on the host. They just need whitelisting; the tools must be Flex-level to resolve.
    assert "ReAct" in bridge.BRIDGEABLE_KINDS and "ReActV2" in bridge.BRIDGEABLE_KINDS
    flex = Flex(ShoutSig, tools=[shout], interpreter_factory=lambda: MockInterpreter())
    tools_payload = [{bridge.TOOL_MARKER: "shout"}]
    flex._bridge._construct("ReAct", "text: str -> out: str", "agent", {"tools": tools_payload})
    assert isinstance(flex.agent, dspy.ReAct)
    # No interpreter is injected into a non-code-executing predictor.
    assert not hasattr(flex.agent, "interpreter")


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
    flex = Flex(Doubler, tools=[shout], interpreter_factory=lambda: MockInterpreter())
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
# Code-executing sub-predictors inherit the Flex sandbox backend (no Deno)
# =============================================================================
# A bridged CodeAct/ProgramOfThought/RLM runs its OWN inner code. It must run that code in the
# sandbox backend chosen for Flex (via the Flex interpreter factory), not a separate default
# Deno/Pyodide sandbox — otherwise sandboxing the Flex glue but not the code its sub-predictors
# execute would be a hole.


def test_accepts_interpreter_factory_only_for_code_executing_kinds() -> None:
    # Introspection-based, not a hardcoded list, so it tracks the real constructors.
    assert bridge._accepts_interpreter_factory(dspy.CodeAct)
    assert bridge._accepts_interpreter_factory(dspy.ProgramOfThought)
    assert bridge._accepts_interpreter_factory(dspy.RLM)  # type: ignore[arg-type]  # dspy re-exports RLM oddly
    # Pure-LM and tool-calling predictors run no inner code, so no interpreter factory is injected.
    assert not bridge._accepts_interpreter_factory(dspy.Predict)
    assert not bridge._accepts_interpreter_factory(dspy.ChainOfThought)
    assert not bridge._accepts_interpreter_factory(dspy.ReAct)
    assert not bridge._accepts_interpreter_factory(dspy.ReActV2)  # type: ignore[arg-type]  # re-export quirk


def test_bridged_codeact_inherits_flex_interpreter_backend() -> None:
    # CodeAct requires function tools; `shout` is a Flex-level tool, resolvable on the host.
    flex = Flex(ShoutSig, tools=[shout], interpreter_factory=lambda: MockInterpreter())
    tools_payload = [{bridge.TOOL_MARKER: "shout"}]  # how the shim passes a tool by name
    flex._bridge._construct("CodeAct", "text: str -> out: str", "act", {"tools": tools_payload})
    # The sub-predictor makes its per-forward interpreter from the Flex factory (the configured
    # backend), not a default PythonInterpreter.
    assert isinstance(flex.act, dspy.CodeAct)
    assert flex.act._interpreter_factory is flex._bridge._factory
    assert isinstance(flex.act._interpreter_factory(), MockInterpreter)


def test_sub_interpreter_factory_makes_a_fresh_interpreter_per_call() -> None:
    flex = _bridged_flex()  # a real factory (lambda: MockInterpreter())
    factory = flex._bridge._sub_interpreter_factory()
    assert factory is flex._bridge._factory  # the sub-predictor gets the Flex factory itself
    a, b = factory(), factory()
    assert isinstance(a, MockInterpreter) and isinstance(b, MockInterpreter)
    assert a is not b  # so each forward is isolated in its own interpreter


# =============================================================================
# Integration (Deno) — the real run-in-interpreter bridge
# =============================================================================

# An optimized CoT module with optimizer-authored glue: the ChainOfThought call must bridge to the
# host, while the import/regex/int parsing run in the sandbox.
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
    flex = Flex(Doubler, interpreter_factory=lambda: dspy.PythonInterpreter())
    try:
        flex._bind_code(COT_GLUE_MODULE)
        # forward() runs entirely in the sandbox; self.solve(...) bridges to the host DummyLM, and the
        # `import re` / int-parsing glue executes inside Pyodide.
        out = flex(value=21)
        assert isinstance(out, dspy.Prediction)
        assert out.result == 42
        # The predictor is a real host object discoverable for optimization/serialization.
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
    program = Flex(Doubler, interpreter_factory=lambda: dspy.PythonInterpreter())
    path = tmp_path / "prog.json"
    try:
        program._bind_code(COT_GLUE_MODULE)
        program.save(path)
    finally:
        program.close()

    # The interpreter is a runtime dependency (like the LM): the saved code persists, the sandbox does not.
    assert "interpreter" not in path.read_text()

    # Reconstruct with a fresh interpreter factory and load the saved code.
    reloaded = Flex(Doubler, interpreter_factory=lambda: dspy.PythonInterpreter())
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
    flex = Flex(ShoutSig, tools=[shout], interpreter_factory=lambda: dspy.PythonInterpreter())
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
    flex = Flex(Doubler, interpreter_factory=lambda: dspy.PythonInterpreter(), max_predictor_calls=3)
    try:
        flex._bind_code(RUNAWAY_MODULE)
        with pytest.raises(Exception) as excinfo:  # the budget error propagates out of the sandbox
            flex(value=1)
        assert "budget" in str(excinfo.value).lower()
    finally:
        flex.close()


# A module that constructs and calls a dspy.CodeAct with a Flex-level tool. CodeAct runs its own
# LM-authored code in an interpreter; under the bridge that interpreter is a fresh instance from the
# Flex factory, so CodeAct's code runs in the SAME sandbox backend configured for Flex.
CODEACT_MODULE = textwrap.dedent(
    """
    class ShoutModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.act = dspy.CodeAct("text: str -> out: str", tools=[shout])

        def forward(self, **inputs):
            r = self.act(text=inputs["text"])
            return dspy.Prediction(out=r.out)
    """
).strip()


@deno_required
def test_bridged_codeact_runs_in_flex_sandbox() -> None:
    # CodeAct generates code that calls the (Flex-level) `shout` tool, then extracts the answer.
    dspy.configure(
        lm=DummyLM(
            [
                {"generated_code": "```python\nprint(shout('hello'))\n```", "finished": True},
                {"reasoning": "the code printed HELLO", "out": "HELLO"},
            ]
        )
    )

    def backend_factory():
        return dspy.PythonInterpreter()

    flex = Flex(ShoutSig, tools=[shout], interpreter_factory=backend_factory)
    try:
        flex._bind_code(CODEACT_MODULE)
        out = flex(text="hello")
        assert out.out == "HELLO"
        # The sub-CodeAct inherited the Flex-configured backend rather than a default PythonInterpreter:
        # it builds its per-forward interpreter from the very factory Flex was given.
        act = flex.act  # bridged predictor, attached dynamically
        assert act._interpreter_factory is backend_factory
    finally:
        flex.close()


# A module that constructs and calls a dspy.ReAct with a Flex-level tool. ReAct runs no code of its
# own: the LM picks a tool, the host runs it, and the LM extracts the answer — all bridged from the
# sandbox where the module's glue runs.
REACT_MODULE = textwrap.dedent(
    """
    class ShoutModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.agent = dspy.ReAct("text: str -> out: str", tools=[shout])

        def forward(self, **inputs):
            r = self.agent(text=inputs["text"])
            return dspy.Prediction(out=r.out)
    """
).strip()


@deno_required
def test_bridged_react_runs_in_flex_sandbox() -> None:
    dspy.configure(
        lm=DummyLM(
            [
                {"next_thought": "shout it", "next_tool_name": "shout", "next_tool_args": {"text": "hello"}},
                {"next_thought": "done", "next_tool_name": "finish", "next_tool_args": {}},
                {"reasoning": "the tool returned HELLO", "out": "HELLO"},
            ]
        )
    )
    flex = Flex(ShoutSig, tools=[shout], interpreter_factory=lambda: dspy.PythonInterpreter())
    try:
        flex._bind_code(REACT_MODULE)
        out = flex(text="hello")
        # The agent's glue ran in the sandbox; the `shout` tool call and every LM call bridged to the host.
        assert out.out == "HELLO"
        assert isinstance(flex.agent, dspy.ReAct)
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

    flex = Flex(Leak, interpreter_factory=lambda: dspy.PythonInterpreter())
    try:
        flex._bind_code(_evil_read_module(str(secret)))
        out = flex(value=1)
        # The sandbox could not reach the host file; the secret never crossed the boundary.
        assert "TOP-SECRET" not in str(out.result)
        assert str(out.result).startswith("BLOCKED")
    finally:
        flex.close()


@deno_required
def test_nested_rlm_tool_provenance() -> None:
    """A GEPA-authored RLM spawns a second RLM through a host tool, and a host-provided tool is
    threaded down to both. Assert the trust boundary AND that each layer is a distinct interpreter:

      - each layer runs in a Pyodide sandbox (its ``sys.platform`` is 'emscripten', not the host's);
      - each layer is its OWN interpreter, not one shared sandbox. A global counter persists WITHIN
        an interpreter but resets in a fresh one, so the Flex glue counts 1 then 2 in interp A, while
        each nested agent sees 1 (a fresh namespace) — if they shared the glue's interpreter it would
        read 3, then 4;
      - the host-provided ``probe`` tool, wherever it is called from, executes in the HOST process.

    A code-executing agent's REPL has no ``dspy`` shim, so the second RLM is spawned via a host tool
    (``spawn_inner``) that constructs and runs it on the host, passing ``probe`` further down; only
    that sub-agent's ``out`` string crosses back, so its trajectory never touches the Flex bridge.
    """
    import sys as _sys

    host_platform = _sys.platform
    assert host_platform != "emscripten"  # sanity: the test itself is the real host, not the sandbox
    record: dict[str, dict[str, str]] = {}
    host_seen: dict[str, str] = {}

    def probe(layer: str, platform: str, count: str) -> str:
        # Host-provided tool: runs in the host process. Records, per layer, the platform and the
        # per-interpreter counter value that layer reported, plus the platform `probe` itself runs in.
        record[layer] = {"platform": platform, "count": count}
        host_seen["platform"] = __import__("sys").platform
        return "ok"

    def spawn_inner(task: str) -> str:
        # Host tool that spawns a SECOND code-executing agent (a nested interpreter), passing the
        # host `probe` tool down to it. Runs on the host; only the inner RLM's `out` string is
        # returned, so nothing but a string crosses back.
        inner = dspy.RLM("task: str -> out: str", tools=[probe])
        return inner(task=task).out

    class ProbeSig(dspy.Signature):
        task: str = dspy.InputField()
        out: str = dspy.OutputField()

    # Each agent bumps a global counter and reports it; a fresh interpreter starts the count at 1.
    outer_code = (
        "```python\nimport sys\ng = globals()\ng['_n'] = g.get('_n', 0) + 1\n"
        "probe(layer='outer_rlm', platform=sys.platform, count=str(g['_n']))\n"
        "spawn_inner(task='go')\nSUBMIT(out='outer')\n```"
    )
    inner_code = (
        "```python\nimport sys\ng = globals()\ng['_n'] = g.get('_n', 0) + 1\n"
        "probe(layer='inner_rlm', platform=sys.platform, count=str(g['_n']))\nSUBMIT(out='inner')\n```"
    )
    dspy.configure(
        lm=DummyLM(
            [
                {"reasoning": "probe here, then spawn the sub-agent", "code": outer_code},
                {"reasoning": "probe from the sub-agent", "code": inner_code},
            ]
        )
    )

    MODULE = textwrap.dedent(
        """
        class ProbeModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.rlm = dspy.RLM("task: str -> out: str", tools=[probe, spawn_inner])

            def forward(self, **inputs):
                import sys
                g = globals()
                g["_n"] = g.get("_n", 0) + 1                                        # interp A -> 1
                probe(layer="glue_1", platform=sys.platform, count=str(g["_n"]))
                g["_n"] = g.get("_n", 0) + 1                                        # same interp -> 2
                probe(layer="glue_2", platform=sys.platform, count=str(g["_n"]))
                self.rlm(task=inputs["task"])
                return dspy.Prediction(out="done")
        """
    ).strip()

    with Flex(ProbeSig, tools=[probe, spawn_inner], interpreter_factory=lambda: dspy.PythonInterpreter()) as flex:
        flex._bind_code(MODULE)
        flex(task="go")

    # Every layer ran inside a Pyodide sandbox, not the host process:
    for layer in ("glue_1", "glue_2", "outer_rlm", "inner_rlm"):
        assert record[layer]["platform"] == "emscripten", f"Failed at {layer}."
    # The counter persists WITHIN the Flex glue's interpreter (1 then 2)...
    assert record["glue_1"]["count"] == "1"
    assert record["glue_2"]["count"] == "2"
    # ...but each nested agent starts fresh at 1 — proving it is its OWN interpreter, not the glue's
    # (which would read 3, then 4) and not each other's:
    assert record["outer_rlm"]["count"] == "1"
    assert record["inner_rlm"]["count"] == "1"
    # The host-provided tool itself executed in the host process, wherever it was called from:
    assert host_seen["platform"] == host_platform
