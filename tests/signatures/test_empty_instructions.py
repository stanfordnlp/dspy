"""Regression tests for https://github.com/stanfordnlp/dspy/issues/10908

Empty signature instructions should remain empty, not be replaced with
the default "Given the fields ... produce the fields ..." text.
"""

import dspy


def test_empty_string_instructions_preserved():
    sig = dspy.Signature("a, b -> c", "")
    assert sig.instructions == ""


def test_with_instructions_empty_string():
    sig = dspy.Signature("a, b -> c", "Some instructions")
    empty_sig = sig.with_instructions("")
    assert empty_sig.instructions == ""


def test_none_instructions_still_get_defaults():
    sig = dspy.Signature("a, b -> c")
    assert "Given the fields" in sig.instructions


def test_normal_instructions_unchanged():
    sig = dspy.Signature("a, b -> c", "Translate to French.")
    assert sig.instructions == "Translate to French."
