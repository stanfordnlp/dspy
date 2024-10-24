import copy
import textwrap

import pydantic
import pytest
import ujson

import dspy
from dspy import Predict, Signature, TypedPredictor
from dspy.utils.dummies import DSPDummyLM


def test_call_method():
    predict_instance = Predict("input -> output")
    lm = DSPDummyLM(["test output"])
    dspy.settings.configure(lm=lm)
    result = predict_instance(input="test input")
    assert result.output == "test output"
    assert lm.get_convo(-1) == (
        "Given the fields `input`, produce the fields `output`.\n"
        "\n---\n\n"
        "Follow the following format.\n\n"
        "Input: ${input}\n"
        "Output: ${output}\n"
        "\n---\n\n"
        "Input: test input\n"
        "Output: test output"
    )


def test_forward_method():
    program = Predict("question -> answer")
    dspy.settings.configure(lm=DSPDummyLM([]))
    result = program(question="What is 1+1?").answer
    assert result == "No more responses"


def test_forward_method2():
    program = Predict("question -> answer1, answer2")
    dspy.settings.configure(lm=DSPDummyLM(["my first answer", "my second answer"]))
    result = program(question="What is 1+1?")
    assert result.answer1 == "my first answer"
    assert result.answer2 == "my second answer"


def test_config_management():
    predict_instance = Predict("input -> output")
    predict_instance.update_config(new_key="value")
    config = predict_instance.get_config()
    assert "new_key" in config and config["new_key"] == "value"


def test_multi_output():
    program = Predict("question -> answer", n=2)
    dspy.settings.configure(lm=DSPDummyLM(["my first answer", "my second answer"]))
    results = program(question="What is 1+1?")
    assert results.completions.answer[0] == "my first answer"
    assert results.completions.answer[1] == "my second answer"


def test_multi_output2():
    program = Predict("question -> answer1, answer2", n=2)
    dspy.settings.configure(
        lm=DSPDummyLM(
            [
                "my 0 answer\nAnswer 2: my 2 answer",
                "my 1 answer\nAnswer 2: my 3 answer",
            ],
        )
    )
    results = program(question="What is 1+1?")
    assert results.completions.answer1[0] == "my 0 answer"
    assert results.completions.answer1[1] == "my 1 answer"
    assert results.completions.answer2[0] == "my 2 answer"
    assert results.completions.answer2[1] == "my 3 answer"


def test_output_only():
    class OutputOnlySignature(dspy.Signature):
        output = dspy.OutputField()

    predictor = Predict(OutputOnlySignature)

    lm = DSPDummyLM(["short answer"])
    dspy.settings.configure(lm=lm)
    assert predictor().output == "short answer"

    assert lm.get_convo(-1) == textwrap.dedent(
        """\
        Given the fields , produce the fields `output`.

        ---

        Follow the following format.

        Output: ${output}

        ---

        Output: short answer"""
    )
