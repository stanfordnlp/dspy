"""dspy.Flex accepts tools, hands them to the underlying RLM baseline, exposes them to the
GEPA code proposer, and round-trips them through save/load.

Tools are callables/dspy.Tool objects — they are NOT serialized. Like the signature, they are
re-provided at construction; the saved ``module_src`` references them by name and resolves
against the reconstructed Flex's context at bind time.
"""

from __future__ import annotations

import textwrap

import pytest

import dspy
from dspy.flex import Flex


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


def test_baseline_rlm_gets_the_tools() -> None:
    program = Flex(Echo, tools=[shout])
    assert "dspy.RLM(" in program.module_src
    assert "tools=[shout]" in program.module_src


def test_no_tools_means_no_tools_arg() -> None:
    assert "tools=[" not in Flex(Echo).module_src


def test_tool_is_in_scope_for_bound_code() -> None:
    program = Flex(Echo, tools=[shout])
    program._bind_code(TOOL_MODULE)
    # LM-free: forward only calls the tool.
    assert program(q="hello").a == "HELLO"


def test_save_load_roundtrips_tool_using_code(tmp_path) -> None:
    program = Flex(Echo, tools=[shout])
    program._bind_code(TOOL_MODULE)

    path = tmp_path / "program.json"
    program.save(path)

    # Reconstruct with the SAME tools (the architecture is re-provided in code, like the
    # signature), then load the saved code — the tool reference resolves and runs.
    reloaded = Flex(Echo, tools=[shout])
    reloaded.load(path)
    assert "shout(q)" in reloaded.module_src
    assert reloaded(q="hi").a == "HI"


def test_load_without_tools_cannot_resolve_them(tmp_path) -> None:
    # The baseline RLM is built with tools=[shout] in __init__; reconstructing without the
    # tool can't resolve the name, so binding the saved code raises (tools must be re-provided).
    program = Flex(Echo, tools=[shout])
    path = tmp_path / "program.json"
    program.save(path)

    reloaded = Flex(Echo)  # tools NOT re-provided
    with pytest.raises(NameError):
        reloaded.load(path)


def test_gepa_proposer_is_told_about_tools() -> None:
    """GEPA can optimize tool usage because the proposer is shown the available tools."""
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    def _metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        return 1.0

    adapter = DspyAdapter(student_module=Flex(Echo, tools=[shout]), metric_fn=_metric, feedback_map={})
    assert "shout" in adapter._flex_context_blurbs["self"]


# A module that AUTHORS its own tool (defined in __init__) and hands it to the RLM — what GEPA
# is encouraged to do. The tool lives in the source, so it persists with the code itself.
SELF_AUTHORED_TOOL_MODULE = textwrap.dedent("""
    class EchoModule(dspy.Module):
        def __init__(self):
            super().__init__()
            def upcase(text: str) -> str:
                "Uppercase the text."
                return text.upper()
            self.rlm = dspy.RLM("q: str -> a: str", tools=[upcase])

        def forward(self, **inputs):
            result = self.rlm(**inputs)
            return dspy.Prediction(a=result.a)
""").strip()


def test_self_authored_tool_persists_with_the_code(tmp_path) -> None:
    # Tools GEPA writes itself live in module_src, so they persist exactly like the code (and
    # like instructions): reconstructing with ONLY the signature — no tools re-provided — and
    # loading rebuilds the tool from the saved source.
    flex = Flex(Echo)  # no user-provided tools
    flex._bind_code(SELF_AUTHORED_TOOL_MODULE)
    assert "tools=[upcase]" in flex.module_src

    path = tmp_path / "program.json"
    flex.save(path)

    reloaded = Flex(Echo)  # still no tools provided
    reloaded.load(path)
    assert reloaded.module_src == flex.module_src
    assert hasattr(reloaded, "rlm")  # rebuilt the RLM whose tool is defined in the code


# Instructions baked into the generated code, on an inner predictor.
INSTR_MODULE = textwrap.dedent("""
    class EchoModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.p = dspy.Predict(dspy.Signature("q -> a", "Always answer in uppercase."))

        def forward(self, q):
            return dspy.Prediction(a=self.p(q=q).a)
""").strip()


def test_inner_predictor_instructions_persist(tmp_path) -> None:
    # The instructions GEPA writes into the code persist through save/load (via module_src and
    # the predictor's own serialized state), just like a plain optimized Predict.
    flex = Flex(Echo)
    flex._bind_code(INSTR_MODULE)
    assert flex.p.signature.instructions == "Always answer in uppercase."

    path = tmp_path / "program.json"
    flex.save(path)

    reloaded = Flex(Echo)
    reloaded.load(path)
    assert reloaded.p.signature.instructions == "Always answer in uppercase."
