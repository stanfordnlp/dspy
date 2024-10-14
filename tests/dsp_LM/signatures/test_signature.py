import textwrap
from typing import List

import pydantic
import pytest

import dspy
from dspy import InputField, OutputField, Signature, infer_prefix
from dspy.utils.dummies import DSPDummyLM


def test_multiline_instructions():
    class MySignature(Signature):
        """First line
        Second line
            Third line"""

        output = OutputField()

    predictor = dspy.Predict(MySignature)

    lm = DSPDummyLM(["short answer"])
    dspy.settings.configure(lm=lm)
    assert predictor().output == "short answer"

    assert lm.get_convo(-1) == textwrap.dedent(
        """\
        First line
        Second line
            Third line

        ---

        Follow the following format.

        Output: ${output}

        ---

        Output: short answer"""
    )
