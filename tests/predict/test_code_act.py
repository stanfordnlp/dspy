import pytest
import shutil

import dspy
from dspy import Signature
from dspy.predict import CodeAct
from dspy.utils import DummyLM

# This test suite requires deno to be installed. Please install deno following https://docs.deno.com/runtime/getting_started/installation/
is_deno_available = shutil.which("deno") is not None
skip_if_deno_not_available = pytest.mark.skipif(
    not is_deno_available, reason="Deno is not installed or not in PATH"
)


class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


def add(a: float, b: float) -> float:
    "add two numbers"
    return a + b


@skip_if_deno_not_available
def test_codeact_code_generation():
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = add(1,1)\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.settings.configure(lm=lm)
    program = CodeAct(BasicQA, tools=[add])
    res = program(question="What is 1+1?")
    assert res.answer == "2"
    assert res.trajectory == {
        'code_output_0': '"2\\n"',
        'generated_code_0': 'result = add(1,1)\nprint(result)',
    }
    assert program.interpreter.deno_process is None


class ExtremumFinder(Signature):
    input_list = dspy.InputField()
    maximum = dspy.OutputField(desc="The maximum of the given numbers")
    minimum = dspy.OutputField(desc="The minimum of the given numbers")

def extract_maximum_minimum(input_list: str) -> dict[str, float]:
    numbers = list(map(float, input_list.split(",")))
    return {"maximum": max(numbers), "minimum": min(numbers)}

@skip_if_deno_not_available
def test_codeact_support_multiple_fields():
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = extract_maximum_minimum('2, 3, 5, 6')\nprint(result)\n```",
                "finished": True,
            },
            {"reasoning": "Reason_B", "maximum": "6", "minimum": "2"},
        ]
    )
    dspy.settings.configure(lm=lm)
    program = CodeAct(ExtremumFinder, tools=[extract_maximum_minimum])
    res = program(input_list="2, 3, 5, 6")
    assert res.maximum == "6"
    assert res.minimum == "2"
    assert res.trajectory == {
        'code_output_0': '"{\'maximum\': 6.0, \'minimum\': 2.0}\\n"',
        'generated_code_0': "result = extract_maximum_minimum('2, 3, 5, 6')\nprint(result)",
    }
    assert program.interpreter.deno_process is None

class CustomTool:
    def __call__(self, a: float, b: float) -> float:
        return a + b

@skip_if_deno_not_available
def test_codeact_tool_validation():
    with pytest.raises(ValueError, match="CodeAct only accepts functions and not callable objects."):
        CodeAct(BasicQA, tools=[CustomTool()])
