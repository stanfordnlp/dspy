"""dspy.Flex accepts tools, hands them to the underlying RLM baseline, exposes them to the
GEPA code proposer, and round-trips them through save/load.

Tools are callables/dspy.Tool objects — they are NOT serialized. Like the signature, they are
re-provided at construction; the saved ``module_src`` references them by name and resolves
against the reconstructed Flex's context at bind time.

Flex runs its code through an interpreter (no in-process mode), so tests that only inspect
``module_src`` / context use a MockInterpreter (Deno-free), while tests that bind-and-run
generated code are marked ``deno_required``.
"""

from __future__ import annotations

import shutil
import textwrap

import pytest

import dspy
from dspy.flex import Flex
from dspy.primitives.code_interpreter import CodeInterpreterError
from tests.mock_interpreter import MockInterpreter

deno_required = pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed")


def shout(text: str) -> str:
    """Return the text in uppercase."""
    return text.upper()


class Echo(dspy.Signature):
    """Echo the question as the answer."""

    q: str = dspy.InputField()
    a: str = dspy.OutputField()


# A tool-using module with NO predictors: forward calls the in-scope `shout` tool directly,
# so it runs LM-free (stands in for code GEPA might produce that wires a tool in).
TOOL_MODULE = textwrap.dedent("""
    class EchoModule(dspy.Module):
        def __init__(self):
            super().__init__()

        def forward(self, q):
            return dspy.Prediction(a=shout(q))
""").strip()


def test_baseline_is_rlm_with_tools() -> None:
    # With tools, the baseline delegates to dspy.RLM (which can call them via its REPL), not Predict,
    # and the tools are wired into its constructor by name.
    src = Flex(Echo, tools=[shout], interpreter_factory=lambda: MockInterpreter()).module_src
    assert "dspy.RLM(" in src
    assert "tools=[shout]" in src
    assert "dspy.Predict(" not in src


def test_baseline_is_predict_without_tools() -> None:
    # Without tools, the baseline is a single dspy.Predict (no RLM, no tools arg).
    src = Flex(Echo, interpreter_factory=lambda: MockInterpreter()).module_src
    assert "dspy.Predict(" in src
    assert "dspy.RLM(" not in src
    assert "tools=[" not in src


@deno_required
def test_tool_is_in_scope_for_bound_code() -> None:
    with Flex(Echo, tools=[shout], interpreter_factory=lambda: dspy.PythonInterpreter()) as program:
        program._bind_code(TOOL_MODULE)
        # LM-free: forward only calls the tool (bridged from the sandbox to the host).
        assert program(q="hello").a == "HELLO"


@deno_required
def test_save_load_roundtrips_tool_using_code(tmp_path) -> None:
    path = tmp_path / "program.json"
    with Flex(Echo, tools=[shout], interpreter_factory=lambda: dspy.PythonInterpreter()) as program:
        program._bind_code(TOOL_MODULE)
        program.save(path)

    # Reconstruct with the SAME tools (the architecture is re-provided in code, like the
    # signature), then load the saved code — the tool reference resolves and runs.
    with Flex(Echo, tools=[shout], interpreter_factory=lambda: dspy.PythonInterpreter()) as reloaded:
        reloaded.load(path)
        assert "shout(q)" in reloaded.module_src
        assert reloaded(q="hi").a == "HI"


@deno_required
def test_load_without_tools_cannot_resolve_them(tmp_path) -> None:
    # The baseline RLM is built with tools=[shout] in __init__; reconstructing without the
    # tool can't resolve the name in the sandbox, so binding the saved code raises (tools must be
    # re-provided).
    path = tmp_path / "program.json"
    with Flex(Echo, tools=[shout], interpreter_factory=lambda: dspy.PythonInterpreter()) as program:
        program.save(path)

    with Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter()) as reloaded:  # tools NOT re-provided
        with pytest.raises(CodeInterpreterError):
            reloaded.load(path)


def test_gepa_proposer_is_told_about_tools() -> None:
    """GEPA can optimize tool usage because the proposer is shown the available tools."""
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    def _metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return 1.0

    student = Flex(Echo, tools=[shout], interpreter_factory=lambda: MockInterpreter())
    adapter = DspyAdapter(student_module=student, metric_fn=_metric, feedback_map={})
    assert "shout" in adapter._flex_context_blurbs["self"]


# A module that AUTHORS its own helper (defined inside forward) and calls it directly — what GEPA is
# encouraged to do in a sandbox, where an authored function cannot be handed to a bridged
# sub-predictor. The helper lives in the source, so it persists with the code itself.
SELF_AUTHORED_TOOL_MODULE = textwrap.dedent("""
    class EchoModule(dspy.Module):
        def __init__(self):
            super().__init__()

        def forward(self, **inputs):
            def upcase(text: str) -> str:
                "Uppercase the text."
                return text.upper()
            return dspy.Prediction(a=upcase(inputs["q"]))
""").strip()


@deno_required
def test_self_authored_tool_persists_with_the_code(tmp_path) -> None:
    # Helpers GEPA writes itself live in module_src, so they persist exactly like the code (and like
    # instructions): reconstructing with ONLY the signature — no tools re-provided — rebuilds the
    # helper from the saved source and runs it.
    path = tmp_path / "program.json"
    with Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter()) as flex:  # no user-provided tools
        flex._bind_code(SELF_AUTHORED_TOOL_MODULE)
        assert "def upcase" in flex.module_src
        flex.save(path)

    with Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter()) as reloaded:  # still no tools
        reloaded.load(path)
        assert reloaded.module_src == SELF_AUTHORED_TOOL_MODULE
        assert reloaded(q="hi").a == "HI"


# Instructions baked into the generated code, on an inner predictor.
INSTR_MODULE = textwrap.dedent("""
    class EchoModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.p = dspy.Predict(dspy.Signature("q -> a", "Always answer in uppercase."))

        def forward(self, q):
            return dspy.Prediction(a=self.p(q=q).a)
""").strip()


@deno_required
def test_inner_predictor_instructions_persist(tmp_path) -> None:
    # The instructions GEPA writes into the code persist through save/load (via module_src and
    # the predictor's own serialized state), just like a plain optimized Predict.
    path = tmp_path / "program.json"
    with Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter()) as flex:
        flex._bind_code(INSTR_MODULE)
        assert flex.p.signature.instructions == "Always answer in uppercase."
        flex.save(path)

    with Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter()) as reloaded:
        reloaded.load(path)
        assert reloaded.p.signature.instructions == "Always answer in uppercase."
