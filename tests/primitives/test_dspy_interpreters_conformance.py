"""Conformance against dspy-interpreters, the third-party CodeInterpreter backends (a test-only
dependency, declared in the ``dev`` dependency group).

Both directions of the capability contract are pinned on real backends: undeclared backends get no
sub-agents, a subclass declaring ``InterpreterCapability.SUB_DSPY`` hosts the facade and runs bridged
sub-agents, and the library's own consumer checks pass.
"""

import pytest

import dspy
from dspy.predict.rlm import RLM
from dspy.primitives.code_interpreter import InterpreterCapability, interpreter_capabilities
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM

dspy_interpreters = pytest.importorskip("dspy_interpreters")

InProcessInterpreter = dspy_interpreters.InProcessInterpreter
SubprocessInterpreter = dspy_interpreters.SubprocessInterpreter
BACKENDS = ["InProcessInterpreter", "SubprocessInterpreter"]


class FacadeSubprocessInterpreter(SubprocessInterpreter):
    capabilities = InterpreterCapability.SUB_DSPY


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
def test_backends_satisfy_protocol_and_declare_no_capabilities(backend_name):
    backend = getattr(dspy_interpreters, backend_name)
    interpreter = backend()
    try:
        assert isinstance(interpreter, dspy.CodeInterpreter)
    finally:
        interpreter.shutdown()
    assert not interpreter_capabilities(backend)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_undeclared_backends_get_no_sub_agents(backend_name):
    rlm = RLM("query -> answer", interpreter_factory=getattr(dspy_interpreters, backend_name))
    assert rlm._sub_dspy is False
    assert "Sub-agents (dspy)" not in rlm.generate_action.signature.instructions


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_library_consumer_checks_pass(backend_name):
    backend = getattr(dspy_interpreters, backend_name)
    dspy_interpreters.check_interpreter(backend).raise_for_failures()
    dspy_interpreters.check_rlm(backend).raise_for_failures()


def test_flex_facade_needs_an_isolated_backend():
    dspy_interpreters.check_flex_facade(SubprocessInterpreter).raise_for_failures()
    report = dspy_interpreters.check_flex_facade(InProcessInterpreter)
    assert not report.passed and "host's memory" in report.results[0].detail


def test_declaring_backend_runs_bridged_sub_agents_with_host_tools():
    # Regression: worker backends bind host tools as "<lambda>" proxies; the shim names them by global.
    calls = []

    def echo(text: str) -> str:
        """Echo the text."""
        calls.append(text)
        return text

    rlm = RLM("query -> answer", max_iters=2, tools=[echo], interpreter_factory=FacadeSubprocessInterpreter)
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
