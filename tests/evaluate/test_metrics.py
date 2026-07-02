# FILEPATH: /Users/ahle/repos/dspy/tests/evaluate/test_metrics.py

import dspy
from dspy.evaluate.metrics import answer_exact_match, tool_call_exact_match
from dspy.predict import Predict


def test_answer_exact_match_string():
    example = dspy.Example(
        question="What is 1+1?",
        answer="2",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "2"
    assert answer_exact_match(example, pred)


def test_answer_exact_match_list():
    example = dspy.Example(
        question="What is 1+1?",
        answer=["2", "two"],
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "2"
    assert answer_exact_match(example, pred)


def test_answer_exact_match_no_match():
    example = dspy.Example(
        question="What is 1+1?",
        answer="2",
    ).with_inputs("question")
    pred = Predict("question -> answer")
    pred.answer = "3"
    assert not answer_exact_match(example, pred)


def test_tool_call_exact_match_ignores_call_ids_and_order():
    example = dspy.Example(
        tool_calls=dspy.ToolCalls.from_dict_list(
            [
                {"id": "gold-1", "name": "search", "args": {"query": "DSPy", "limit": 5}},
                {"id": "gold-2", "name": "lookup", "args": {"id": "42"}},
            ]
        )
    )
    pred = dspy.Prediction(
        tool_calls=dspy.ToolCalls.from_dict_list(
            [
                {"id": "pred-2", "name": "lookup", "args": {"id": "42"}},
                {"id": "pred-1", "name": "search", "args": {"query": "DSPy", "limit": 5}},
            ]
        )
    )

    assert tool_call_exact_match(example, pred)


def test_tool_call_exact_match_accepts_provider_shaped_dicts():
    example = dspy.Example(
        tool_calls=[
            {
                "id": "call-gold",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "DSPy"}'},
            }
        ]
    )
    pred = dspy.Prediction(tool_calls=dspy.ToolCalls.from_dict_list([{"name": "search", "args": {"query": "DSPy"}}]))

    assert tool_call_exact_match(example, pred)


def test_tool_call_exact_match_counts_duplicate_calls():
    example = dspy.Example(
        tool_calls=dspy.ToolCalls.from_dict_list(
            [
                {"name": "search", "args": {"query": "DSPy"}},
                {"name": "search", "args": {"query": "DSPy"}},
            ]
        )
    )
    pred = dspy.Prediction(tool_calls=dspy.ToolCalls.from_dict_list([{"name": "search", "args": {"query": "DSPy"}}]))

    assert not tool_call_exact_match(example, pred)


def test_tool_call_exact_match_requires_exact_argument_values():
    example = dspy.Example(tool_calls=dspy.ToolCalls.from_dict_list([{"name": "search", "args": {"limit": 5}}]))
    pred = dspy.Prediction(tool_calls=dspy.ToolCalls.from_dict_list([{"name": "search", "args": {"limit": "5"}}]))

    assert not tool_call_exact_match(example, pred)


def test_tool_call_exact_match_is_exported_from_evaluate():
    from dspy.evaluate import tool_call_exact_match as exported_metric

    example = dspy.Example(tool_calls=dspy.ToolCalls.from_dict_list([{"name": "search", "args": {"query": "DSPy"}}]))
    pred = dspy.Prediction(tool_calls=dspy.ToolCalls.from_dict_list([{"name": "search", "args": {"query": "DSPy"}}]))

    assert exported_metric(example, pred)
