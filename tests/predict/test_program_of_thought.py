from unittest.mock import patch

import pytest

import dspy
from dspy import ProgramOfThought, Signature
from dspy.utils import DummyLM


class BasicQA(Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


@pytest.mark.deno
def test_pot_code_generation():
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = 1+1\nSUBMIT({'answer': result})\n```",
            },
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(BasicQA)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"
    assert pot.interpreter.deno_process is None


# This test ensures the old finetuned saved models still work
@pytest.mark.deno
def test_old_style_pot():
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\nresult = 1+1\n```"},
            {"reasoning": "Reason_B", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(BasicQA)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"
    assert pot.interpreter.deno_process is None


class ExtremumFinder(Signature):
    input_list = dspy.InputField()
    maximum = dspy.OutputField(desc="The maximum of the given numbers")
    minimum = dspy.OutputField(desc="The minimum of the given numbers")


@pytest.mark.deno
def test_pot_support_multiple_fields():
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nmaximum = 6\nminimum = 2\nSUBMIT({'maximum': maximum, 'minimum': minimum})\n```",
            },
            {"reasoning": "Reason_B", "maximum": "6", "minimum": "2"},
        ]
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(ExtremumFinder)
    res = pot(input_list="2, 3, 5, 6")
    assert res.maximum == "6"
    assert res.minimum == "2"
    assert pot.interpreter.deno_process is None


@pytest.mark.deno
def test_pot_code_generation_with_one_error():
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = 1+0/0\nSUBMIT({'answer': result})\n```",
            },
            {
                "reasoning": "Reason_B",
                "generated_code": "```python\nresult = 1+1\nSUBMIT({'answer': result})\n```",
            },
            {"reasoning": "Reason_C", "answer": "2"},
        ]
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(BasicQA)
    res = pot(question="What is 1+1?")
    assert res.answer == "2"
    assert pot.interpreter.deno_process is None


@pytest.mark.deno
def test_pot_code_generation_persistent_errors():
    max_iters = 3
    lm = DummyLM(
        [
            {
                "reasoning": "Reason_A",
                "generated_code": "```python\nresult = 1+0/0\nSUBMIT({'answer': result})\n```",
            },
        ]
        * max_iters
    )
    dspy.configure(lm=lm)

    pot = ProgramOfThought(BasicQA, max_iters=max_iters)
    with pytest.raises(RuntimeError, match="Max hops reached. Failed to run ProgramOfThought: ZeroDivisionError:"):
        pot(question="What is 1+1?")


def test_pot_code_parse_error():
    max_iters = 3
    lm = DummyLM(
        [
            {"reasoning": "Reason_A", "generated_code": "```python\ninvalid=python=code\n```"},
        ]
        * max_iters
    )
    dspy.configure(lm=lm)
    pot = ProgramOfThought(BasicQA, max_iters=max_iters)
    with (
        patch("dspy.predict.program_of_thought.ProgramOfThought._execute_code") as mock_execute_code,
        pytest.raises(
            RuntimeError, match="Max hops reached. Failed to run ProgramOfThought: Error: Code format is not correct."
        ),
    ):
        pot(question="What is 1+1?")
    mock_execute_code.assert_not_called()


# ---------------------------------------------------------------------------
# Regression tests for #9214: _parse_code must not corrupt valid Python
# ---------------------------------------------------------------------------


def _make_pot():
    """Return a ProgramOfThought instance without requiring a real LM."""
    lm = DummyLM([])
    dspy.configure(lm=lm)
    return ProgramOfThought(BasicQA)


def test_parse_code_preserves_escape_sequences_in_strings():
    """Literal \\n inside a string must not be converted to a real newline.

    Previously `.replace("\\\\n", "\\n")` was applied globally, so
    `print(f"\\\\nTotal: {x}")` became `print(f"\\nTotal: {x}")` with a
    literal line-break inside the f-string, producing a SyntaxError.
    """
    pot = _make_pot()
    code_with_escape = 'result = "line1\\nline2"'
    code_data = {"generated_code": f"```python\n{code_with_escape}\n```"}
    parsed, error = pot._parse_code(code_data)
    assert error is None
    # The escape sequence must survive unchanged
    assert "\\n" in parsed, f"Escape sequence was corrupted; got: {parsed!r}"
    assert "\n" not in parsed.split("=", 1)[1], (
        "A real newline was inserted inside the string literal"
    )


def test_parse_code_preserves_equals_inside_strings():
    """A '=' inside a string literal in multi-line code must not trigger
    spurious line-splitting.

    Previously the context-unaware regex on the else-branch matched '='
    inside string literals and inserted newlines mid-string, producing
    broken Python like `data = "users: Alice\\n=25"`.
    """
    pot = _make_pot()
    # Multi-line code: the LLM already emits proper newlines, so only the
    # regex branch (now removed) could corrupt the second line.
    code_lines = 'data = "users: Alice=25, Bob=30"\nresult = len(data)'
    code_data = {"generated_code": f"```python\n{code_lines}\n```"}
    parsed, error = pot._parse_code(code_data)
    assert error is None, f"Unexpected parse error: {error}"
    # The string content must be intact — no spurious newlines injected
    assert '"users: Alice=25, Bob=30"' in parsed, (
        f"String content was corrupted by regex splitting; got: {parsed!r}"
    )


def test_parse_code_extracts_bare_code_without_markdown():
    """Code supplied without a markdown fence must be returned as-is."""
    pot = _make_pot()
    raw = "result = 1 + 1"
    code_data = {"generated_code": raw}
    parsed, error = pot._parse_code(code_data)
    assert error is None
    assert "result = 1 + 1" in parsed


def test_parse_code_appends_result_variable_for_capture():
    """When the last line is a bare assignment, the variable name should be
    appended so the interpreter can capture and return its value."""
    pot = _make_pot()
    code_data = {"generated_code": "```python\nx = 10\nresult = x * 2\n```"}
    parsed, error = pot._parse_code(code_data)
    assert error is None
    lines = [l for l in parsed.splitlines() if l.strip()]
    assert lines[-1].strip() == "result", (
        f"Expected last line to be 'result' for interpreter capture; got {lines[-1]!r}"
    )
