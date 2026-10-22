"""`dspy.settings.interpreter_factory` replaces the default interpreter for code-executing modules.

A module's explicit factory wins, including PythonInterpreter (Deno/Pyodide).
Otherwise the active setting is resolved lazily, falling back to LocalInterpreter.
"""

import re
import warnings

import pytest

import dspy
from dspy.predict.flex.flex import Flex
from dspy.primitives.code_interpreter import (
    FinalOutput,
    _create_interpreter,
    resolve_interpreter_factory,
)
from dspy.primitives.local_interpreter import LocalInterpreter
from dspy.primitives.python_interpreter import PythonInterpreter
from tests.mock_interpreter import MockInterpreter, MockInterpreterFactory


class Doubler(dspy.Signature):
    value: int = dspy.InputField()
    result: int = dspy.OutputField()


class BasicQA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class StaticPredictor:
    def __init__(self, **fields):
        self.fields = fields

    def __call__(self, **kwargs):
        return dspy.Prediction(**self.fields)


def _program_of_thought(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return dspy.ProgramOfThought(BasicQA, **kwargs)


def _code_act(**kwargs):
    def add(a: float, b: float) -> float:
        """add two numbers"""
        return a + b

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return dspy.CodeAct(BasicQA, tools=[add], **kwargs)


# =============================================================================
# resolve_interpreter_factory
# =============================================================================


def test_local_interpreter_is_the_unconfigured_default():
    assert dspy.settings.interpreter_factory is None
    assert resolve_interpreter_factory() is LocalInterpreter


def test_configured_factory_is_used_when_the_caller_has_none():
    factory = MockInterpreterFactory()
    dspy.configure(interpreter_factory=factory)

    assert resolve_interpreter_factory(None) is factory
    assert isinstance(_create_interpreter(None), MockInterpreter)


def test_explicit_deno_factory_wins_over_setting():
    configured = MockInterpreterFactory()
    dspy.configure(interpreter_factory=configured)

    assert isinstance(_create_interpreter(PythonInterpreter), PythonInterpreter)
    assert resolve_interpreter_factory(MockInterpreter) is MockInterpreter


def test_context_scopes_the_choice_and_restores_it():
    outer = MockInterpreterFactory()
    inner = MockInterpreterFactory()
    dspy.configure(interpreter_factory=outer)

    with dspy.context(interpreter_factory=inner):
        assert resolve_interpreter_factory() is inner

    assert resolve_interpreter_factory() is outer


def test_a_configured_interpreter_instance_is_rejected_by_name():
    dspy.configure(interpreter_factory=MockInterpreter())

    with pytest.raises(TypeError, match=re.escape("dspy.settings.interpreter_factory received an object")):
        resolve_interpreter_factory()


def test_a_configured_non_callable_is_rejected_by_name():
    dspy.configure(interpreter_factory=123)

    with pytest.raises(TypeError, match=re.escape("dspy.settings.interpreter_factory must be a zero-argument callable")):
        resolve_interpreter_factory()


# =============================================================================
# The modules that execute code
# =============================================================================


def test_rlm_forward_uses_the_configured_factory():
    factory = MockInterpreterFactory(responses=[FinalOutput({"answer": "42"})])
    dspy.configure(interpreter_factory=factory)

    rlm = dspy.RLM("context -> answer")
    rlm.generate_action = StaticPredictor(reasoning="submit", code='SUBMIT("42")')

    assert rlm(context="ignored").answer == "42"
    assert len(factory.instances) == 1


def test_rlm_reads_execution_instructions_from_the_configured_factory():
    # The initial action prompt includes the instructions from the factory active at construction.
    class Factory(MockInterpreterFactory):
        execution_instructions = "Use this runtime."

    dspy.configure(interpreter_factory=Factory())

    assert "Use this runtime." in dspy.RLM("query -> answer").generate_action.signature.instructions


def test_rlm_refreshes_execution_instructions_for_context_factory():
    class Factory(MockInterpreterFactory):
        def __init__(self, instructions, responses=None):
            super().__init__(responses=responses)
            self.execution_instructions = instructions

    class CapturingAction:
        def __call__(self, **kwargs):
            self.signature = kwargs["signature"]
            return dspy.Prediction(reasoning="submit", code='SUBMIT("42")')

    dspy.configure(interpreter_factory=Factory("global runtime"))
    rlm = dspy.RLM("query -> answer")
    action = CapturingAction()
    rlm.generate_action = action

    context_factory = Factory("context runtime", responses=[FinalOutput({"answer": "42"})])
    with dspy.context(interpreter_factory=context_factory):
        assert rlm(query="ignored").answer == "42"

    assert "context runtime" in action.signature.instructions
    assert "global runtime" not in action.signature.instructions
    assert len(context_factory.instances) == 1


def test_program_of_thought_forward_uses_the_configured_factory():
    factory = MockInterpreterFactory(responses=[FinalOutput({"answer": "2"})])
    dspy.configure(interpreter_factory=factory)

    pot = _program_of_thought()
    pot.code_generate = StaticPredictor(generated_code="SUBMIT({'answer': 2})")
    pot.generate_output = StaticPredictor(answer="2")

    assert pot(question="What is 1+1?").answer == "2"
    assert len(factory.instances) == 1


def test_code_act_forward_uses_the_configured_factory():
    factory = MockInterpreterFactory(responses=["", "2\n"])
    dspy.configure(interpreter_factory=factory)

    program = _code_act()
    program.codeact = StaticPredictor(generated_code="print(add(1, 1))", finished=True)
    program.extractor = StaticPredictor(answer="2")

    assert program(question="What is 1+1?").answer == "2"
    assert len(factory.instances) == 1


def test_flex_forward_uses_the_configured_factory():
    def execute_fn(code, variables):
        # The shim setup and the class definition come with no variables; only the driver
        # code carries the call's inputs, and it is the one that must return the prediction.
        return '{"result": 4}' if variables else ""

    factory = MockInterpreterFactory(execute_fn=execute_fn)
    dspy.configure(interpreter_factory=factory)

    flex = Flex(Doubler)
    flex._bind_code(
        "class Doubling(dspy.Module):\n"
        "    def forward(self, value):\n"
        "        return dspy.Prediction(result=value * 2)\n"
    )

    assert flex(value=2).result == 4
    assert len(factory.instances) == 1


def test_flex_lets_a_sub_predictor_resolve_the_configured_factory_itself():
    factory = MockInterpreterFactory(responses=[FinalOutput({"answer": "42"})])
    flex = Flex(Doubler)
    nested = flex._bridge.invocation()._build_predictor("RLM", "query -> answer", {})
    nested.generate_action = StaticPredictor(reasoning="submit", code='SUBMIT("42")')
    with dspy.context(interpreter_factory=factory):
        assert nested(query="ignored").answer == "42"
    assert len(factory.instances) == 1


@pytest.mark.parametrize(
    "build",
    [
        lambda **kw: dspy.RLM("question -> answer", **kw),
        lambda **kw: Flex(BasicQA, **kw),
        _program_of_thought,
        _code_act,
    ],
)
@pytest.mark.parametrize("explicit", [None, MockInterpreter, PythonInterpreter])
def test_module_factory_precedence_at_execution_time(build, explicit):
    module = build(interpreter_factory=explicit)
    configured = MockInterpreterFactory()
    with dspy.context(interpreter_factory=configured):
        # Resolve at the same boundary used to create each forward's interpreter.
        interpreter = _create_interpreter(module._interpreter_factory)
    assert isinstance(interpreter, explicit or MockInterpreter)
    assert len(configured.instances) == (1 if explicit is None else 0)
