"""Run the shared sub-agent facade on real Monty sessions, mocking only LM responses."""

import asyncio

import pytest

import dspy
from dspy.utils.dummies import DummyLM
from tests.predict.test_rlm import make_mock_predictor

pytest.importorskip("pydantic_monty")


@pytest.mark.parametrize("use_async", [False, True])
def test_standalone_predictors_helpers_and_predictions_persist_across_feeds(use_async):
    rlm = dspy.RLM("query -> answer: int", max_iters=3, interpreter_factory=dspy.MontyInterpreter)
    rlm.generate_action = make_mock_predictor([
        {"reasoning": "Define", "code": (
            "from dspy import Predict as P\n"
            "first = P('question -> first: int')\n"
            "second = P('question -> second: int')\n"
            "def invoke(fn, question): return fn(question=question)\n"
            "result = invoke(first, query)\n"
            "print(result.first)"
        )},
        {"reasoning": "Reuse", "code": "total = result['first'] + second(question=query).second\nprint(total)"},
        {"reasoning": "Submit", "code": "SUBMIT(answer=total + first(question='again').first)"},
    ])
    lm = DummyLM([{"first": 11}, {"second": 23}, {"first": 7}])
    with dspy.context(lm=lm):
        result = asyncio.run(rlm.acall(query="q")) if use_async else rlm(query="q")
    assert result.answer == 41
    assert len(result.trajectory) == 3
    assert result.trajectory[0]["output"] == "11"
    assert result.trajectory[1]["output"] == "34"
    assert len(lm.history) == 3


def test_facade_enforces_options_and_shares_llm_query_budget():
    rlm = dspy.RLM("query -> answer", max_iters=2, max_llm_calls=1, interpreter_factory=dspy.MontyInterpreter)
    rlm.generate_action = make_mock_predictor([
        {"reasoning": "Check restrictions", "code": (
            "try:\n"
            "    dspy.Predict('question -> answer')(question=query, config={'api_key': 'forbidden'})\n"
            "except Exception as e:\n"
            "    print(e)\n"
            "try:\n"
            "    dspy.RLM('question -> answer', interpreter_factory=None)\n"
            "except Exception as e:\n"
            "    print(e)\n"
            "first = llm_query(query)\n"
            "try:\n"
            "    dspy.Predict('question -> answer')(question=query)\n"
            "except Exception as e:\n"
            "    print(e)"
        )},
        {"reasoning": "Submit", "code": "SUBMIT(answer=first)"},
    ])
    lm = DummyLM([{"answer": "one"}, {"answer": "must not be called"}])
    with dspy.context(lm=lm):
        result = rlm(query="q")
    output = result.trajectory[0]["output"]
    assert "may not set LM option(s) ['api_key']" in output
    assert "cannot choose its interpreter_factory" in output
    assert "LLM call limit exceeded: 1 + 1 > 1" in output
    assert "one" in result.answer
    assert len(lm.history) == 1


def test_nested_rlm_inherits_backend_and_supplied_tools():
    sessions = []

    class TrackedMonty(dspy.MontyInterpreter):
        def __init__(self):
            super().__init__()
            sessions.append(self)

    def shout(text: str) -> str:
        return text.upper()

    rlm = dspy.RLM("query -> answer", tools=[shout], max_iters=1, interpreter_factory=TrackedMonty)
    rlm.generate_action = make_mock_predictor([
        {"reasoning": "Delegate", "code": (
            "sentinel = 17\n"
            "sub = dspy.RLM('text -> out', tools=[shout], max_iters=1)\n"
            "result = sub(text=query)\n"
            "SUBMIT(answer=result.out + ':' + str(sentinel))"
        )},
    ])
    with dspy.context(lm=DummyLM([{"reasoning": "Use tool", "code": "sentinel = 29\nSUBMIT(out=shout(text))"}])):
        result = rlm(query="hello")
    assert result.answer == "HELLO:17"
    assert len(result.trajectory) == 1
    assert len(sessions) == 2
    assert all(session._ended for session in sessions)
