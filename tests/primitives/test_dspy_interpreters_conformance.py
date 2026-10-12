"""Conformance against dspy-interpreters, the third-party CodeInterpreter backends (a test-only
dependency, declared in the ``dev`` dependency group).

RLM installs the sandbox dspy facade into every interpreter it creates. On real backends: an isolated
one runs bridged sub-agents, one that executes in the host's memory runs RLM without them, and the
library's own consumer checks pass.
"""

import pytest

import dspy
from dspy.predict.rlm import RLM
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM

dspy_interpreters = pytest.importorskip("dspy_interpreters")

InProcessInterpreter = dspy_interpreters.InProcessInterpreter
SubprocessInterpreter = dspy_interpreters.SubprocessInterpreter
BACKENDS = ["InProcessInterpreter", "SubprocessInterpreter"]


def make_scripted_predictor(responses: list[dict]):
    class ScriptedPredictor:
        def __init__(self):
            self.idx = 0

        def __call__(self, **kwargs):
            response = responses[self.idx % len(responses)]
            self.idx += 1
            return Prediction(**response)

    return ScriptedPredictor()


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_backends_satisfy_protocol(backend_name):
    interpreter = getattr(dspy_interpreters, backend_name)()
    try:
        assert isinstance(interpreter, dspy.CodeInterpreter)
    finally:
        interpreter.shutdown()


def test_in_process_backend_runs_rlm_without_sub_agents(caplog):
    seen = []

    class Recording:
        def __call__(self, signature=None, **kwargs):
            seen.append(signature.instructions)
            return Prediction(reasoning="Submit", code='SUBMIT("ok")')

    rlm = RLM("query -> answer", max_iters=1, interpreter_factory=InProcessInterpreter)
    rlm.generate_action = Recording()
    with caplog.at_level("WARNING", logger="dspy.predict.rlm"):
        assert rlm(query="q").answer == "ok"

    assert "Sub-agents (dspy)" not in seen[0]
    assert "sub-agents are unavailable on InProcessInterpreter" in caplog.text


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_library_consumer_checks_pass(backend_name):
    backend = getattr(dspy_interpreters, backend_name)
    dspy_interpreters.check_interpreter(backend).raise_for_failures()
    dspy_interpreters.check_rlm(backend).raise_for_failures()


def test_flex_facade_needs_an_isolated_backend():
    dspy_interpreters.check_flex_facade(SubprocessInterpreter).raise_for_failures()
    report = dspy_interpreters.check_flex_facade(InProcessInterpreter)
    assert not report.passed and "host's memory" in report.results[0].detail


def test_isolated_backend_runs_bridged_sub_agents_with_host_tools():
    # Regression: worker backends bind host tools as "<lambda>" proxies; the shim names them by global.
    calls = []

    def echo(text: str) -> str:
        """Echo the text."""
        calls.append(text)
        return text

    rlm = RLM("query -> answer", max_iters=2, tools=[echo], interpreter_factory=SubprocessInterpreter)
    rlm.generate_action = make_scripted_predictor([
        {
            "reasoning": "Bridged ReActV2 with a host tool",
            "code": (
                "import dspy\n"
                "agent = dspy.ReActV2('question -> answer', tools=[echo])\n"
                "res = agent(question='hi')\n"
                "print(res.answer)"
            ),
        },
        {"reasoning": "Submit", "code": "SUBMIT(res.answer)"},
    ])
    lm = DummyLM([
        {
            "next_thought": "echo it",
            "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "echo", "args": {"text": "hi"}}]),
        },
        {
            "next_thought": "done",
            "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "echoed hi"}}]),
        },
    ])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = rlm(query="q")

    assert result.answer == "echoed hi"
    assert calls == ["hi"]
