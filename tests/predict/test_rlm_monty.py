"""Run the shared sub-agent facade on real Monty sessions, mocking only LM responses."""

import asyncio
import json

import pytest

import dspy
from dspy.utils.dummies import DummyLM
from tests.predict.test_rlm import _BinarySerializable, _StubSerializable, make_mock_predictor

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


@pytest.mark.parametrize("use_async", [False, True])
def test_native_async_helpers_and_locals_persist(use_async):
    rlm = dspy.RLM("query: int -> answer: int", max_iters=2, interpreter_factory=dspy.MontyInterpreter)
    rlm.generate_action = make_mock_predictor([
        {"reasoning": "Define", "code": (
            "async def compute(value):\n"
            "    increment = 7\n"
            "    return value + locals()['increment']\n"
            "print(await compute(query))"
        )},
        {"reasoning": "Reuse", "code": "SUBMIT(answer=await compute(query + 1))"},
    ])
    result = asyncio.run(rlm.acall(query=6)) if use_async else rlm(query=6)
    assert result.answer == 14
    assert len(result.trajectory) == 2
    assert result.trajectory[0]["output"] == "13"


def test_sandbox_serializable_reconstructs_native_object_once():
    serializations = []

    class Records(dspy.SandboxSerializable):
        def sandbox_setup(self):
            return (
                "import json\n"
                "class Records:\n"
                "    def __init__(self, rows): self.rows = rows\n"
                "    def total(self): return sum(row['value'] for row in self.rows)"
            )

        def to_sandbox(self):
            serializations.append(True)
            return json.dumps([{"value": 2}, {"value": 11}]).encode()

        def sandbox_assignment(self, var_name, data_expr):
            return f"{var_name} = Records(json.loads({data_expr}))"

        def rlm_preview(self, max_chars=500):
            return "Records with rows and a total() method"

    rlm = dspy.RLM("data -> answer: int", max_iters=2, interpreter_factory=dspy.MontyInterpreter)
    rlm.generate_action = make_mock_predictor([
        {"reasoning": "Mutate", "code": "data.rows.append({'value': 7})\nprint(data.total())"},
        {"reasoning": "Reuse", "code": "SUBMIT(answer=data.total())"},
    ])
    result = rlm(data=Records())
    assert result.answer == 20
    assert len(result.trajectory) == 2
    assert result.trajectory[0]["output"] == "20"
    assert serializations == [True]


def test_sandbox_serializable_binary_payload():
    rlm = dspy.RLM("data -> answer: list[int]", max_iters=1, interpreter_factory=dspy.MontyInterpreter)
    rlm.generate_action = make_mock_predictor([
        {"reasoning": "Read binary", "code": "SUBMIT(answer=list(data))"},
    ])
    assert rlm(data=_BinarySerializable()).answer == [255, 254, 253]


def test_external_library_stays_on_host_behind_a_supplied_tool():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"group": ["a", "b", "a"], "value": [2, 7, 11]})
    calls = []

    def group_total(group: str) -> int:
        """Sum the value column for one group in the host DataFrame."""
        calls.append(group)
        return int(frame.loc[frame["group"] == group, "value"].sum())

    rlm = dspy.RLM("group -> answer: int", tools=[group_total], max_iters=1,
                   interpreter_factory=dspy.MontyInterpreter)
    rlm.generate_action = make_mock_predictor([
        {"reasoning": "Query host data", "code": "SUBMIT(answer=group_total(group))"},
    ])
    assert rlm(group="a").answer == 13
    assert calls == ["a"]

    class PandasInput(_StubSerializable):
        def sandbox_setup(self):
            return "import pandas as pd"

    # Installing pandas on the host does not make a pandas-based guest loader work.
    with pytest.raises(dspy.CodeExecutionError, match="No module named 'pandas'"):
        dspy.RLM("data -> answer", interpreter_factory=dspy.MontyInterpreter)(data=PandasInput())
