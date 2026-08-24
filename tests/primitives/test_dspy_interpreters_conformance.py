"""Conformance tests against dspy-interpreters, the third-party CodeInterpreter backends.

The interpreter-capability contract (``capabilities`` / ``InterpreterCapability.SUB_DSPY`` /
``SUB_DSPY_FACTORY_NAME``) is meant to be implemented by interpreter libraries such as
https://github.com/cmpnd-ai/dspy-interpreters (a test-only dependency, declared in the
``dev`` dependency group). These tests pin both directions of that contract against real
backends:

* dspy side: the library's backends satisfy the ``CodeInterpreter`` protocol, a user-written
  subclass can declare ``SUB_DSPY``, and ``dspy.RLM`` then runs real dspy sub-agents inside it.
* library side: a capability-declaring subclass still passes the library's own
  ``check_interpreter`` conformance suite.
"""

import pytest

import dspy
from dspy.predict.rlm import RLM
from dspy.primitives.code_interpreter import SUB_DSPY_FACTORY_NAME, interpreter_capabilities
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM

dspy_interpreters = pytest.importorskip("dspy_interpreters")

InProcessInterpreter = dspy_interpreters.InProcessInterpreter
SubprocessInterpreter = dspy_interpreters.SubprocessInterpreter


class SubDspyInProcessInterpreter(InProcessInterpreter):
    """The user story's reference: a user-written CodeInterpreter that declares its capabilities."""

    capabilities = dspy.InterpreterCapability.SUB_DSPY

    def __init__(self, tools=None, output_fields=None):
        super().__init__(tools=tools, output_fields=output_fields)
        # The sub-dspy contract: the environment provides the nested-interpreter factory.
        self._namespace[SUB_DSPY_FACTORY_NAME] = SubDspyInProcessInterpreter


def make_scripted_predictor(responses: list[dict]):
    class ScriptedPredictor:
        def __init__(self):
            self.idx = 0

        def __call__(self, **kwargs):
            response = responses[self.idx % len(responses)]
            self.idx += 1
            return Prediction(**response)

    return ScriptedPredictor()


@pytest.mark.parametrize("backend_name", ["InProcessInterpreter", "SubprocessInterpreter"])
def test_backends_satisfy_protocol_and_declare_no_capabilities(backend_name):
    backend = getattr(dspy_interpreters, backend_name)
    interpreter = backend()
    try:
        assert isinstance(interpreter, dspy.CodeInterpreter)
    finally:
        interpreter.shutdown()
    assert not interpreter_capabilities(backend)


@pytest.mark.parametrize("backend_name", ["InProcessInterpreter", "SubprocessInterpreter"])
def test_rlm_omits_sub_dspy_guidance_for_undeclared_backends(backend_name):
    # The capability is a declaration, not a detection: these backends could factually run
    # dspy (InProcess shares the host process), but without declaring SUB_DSPY the action
    # prompt must not advertise dspy sub-agents and the sub-dspy setup path must stay off.
    backend = getattr(dspy_interpreters, backend_name)
    rlm = RLM("query -> answer", interpreter_factory=backend)
    instructions = rlm.generate_action.signature.instructions
    assert "Sub-agents (dspy)" not in instructions
    assert "import dspy" not in instructions
    assert SUB_DSPY_FACTORY_NAME not in instructions
    assert rlm._sub_dspy is False


def test_capability_declaring_subclass_passes_library_conformance():
    assert dspy.InterpreterCapability.SUB_DSPY in interpreter_capabilities(SubDspyInProcessInterpreter)
    dspy_interpreters.check_interpreter(SubDspyInProcessInterpreter).raise_for_failures()


def test_subprocess_backend_runs_dspy():
    # "Some interpreters can run dspy": a genuinely separate worker process imports dspy.
    interpreter = SubprocessInterpreter()
    try:
        output = interpreter.execute("import dspy\nprint(dspy.__version__)")
    finally:
        interpreter.shutdown()
    assert dspy.__version__ in str(output)


def test_rlm_runs_dspy_sub_agents_in_capable_backend():
    # End-to-end on a real backend: REPL code imports dspy, runs a dspy.Predict sub-agent
    # against the configured LM, runs a dspy.ReActV2 sub-agent with a REPL-defined tool,
    # and builds a nested dspy.RLM from the environment-provided factory, so each
    # sub-agent gets its own interpreter.
    rlm = RLM("query -> answer", max_iters=4, interpreter_factory=SubDspyInProcessInterpreter)
    rlm.generate_action = make_scripted_predictor([
        {
            "reasoning": "Run a sub-agent",
            "code": (
                "import dspy\n"
                "sub = dspy.Predict('question -> answer')\n"
                "res = sub(question='ping')\n"
                "print(res.answer)"
            ),
        },
        {
            "reasoning": "Run a tool-using ReActV2 sub-agent with a REPL-defined tool",
            "code": (
                "def lookup(query: str) -> str:\n"
                '    """Look things up."""\n'
                "    return f'found {query}'\n"
                "agent = dspy.ReActV2('question -> answer', tools=[lookup])\n"
                "agent_res = agent(question='cats')\n"
                "print(agent_res.answer)"
            ),
        },
        {
            "reasoning": "Nested RLM gets its own interpreter, then submit",
            "code": (
                f"nested = dspy.RLM('q -> a', interpreter_factory={SUB_DSPY_FACTORY_NAME})\n"
                "SUBMIT(res.answer + ' / ' + agent_res.answer)"
            ),
        },
    ])

    lm = DummyLM([
        {"answer": "sub-agent says hi"},
        {
            "next_thought": "look it up",
            "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "lookup", "args": {"query": "cats"}}]),
        },
        {
            "next_thought": "answer now",
            "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "found cats"}}]),
        },
    ])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = rlm(query="q")

    assert result.answer == "sub-agent says hi / found cats"
