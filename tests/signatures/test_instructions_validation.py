import pytest

import dspy
from dspy.signatures.signature import make_signature


class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def test_with_instructions_rejects_a_non_string():
    with pytest.raises(TypeError, match="`instructions` must be a string or None"):
        QA.with_instructions(123)


def test_make_signature_rejects_a_non_string():
    with pytest.raises(TypeError, match="`instructions` must be a string or None"):
        make_signature("question -> answer", instructions=123)


def test_string_instructions_still_work():
    updated = QA.with_instructions("Answer briefly.")

    assert updated.instructions == "Answer briefly."

    assert make_signature("question -> answer", "Be terse.").instructions == "Be terse."
