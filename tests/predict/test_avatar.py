"""Tests for dspy.predict.avatar.Avatar.

Covers:
1. Construction no longer crashes: Avatar and AvatarOptimizer referenced the
   removed `dspy.TypedPredictor` entry point.
2. The `max_iters` budget: the constructor value is honored, a forward-time
   override still wins, and a budget that runs out without a `Finish` action
   still produces the signature's outputs instead of looping forever or
   raising AttributeError.
"""

import dspy
from dspy.predict.avatar.avatar import Avatar
from dspy.predict.avatar.models import Action, Tool
from dspy.teleprompt import AvatarOptimizer


class EchoTool:
    def run(self, query: str) -> str:
        return f"echo:{query}"


class _CountingDummyLM(dspy.utils.DummyLM):
    """DummyLM that fails loudly once it is asked for more answers than expected."""

    def __init__(self, answers, max_calls: int):
        super().__init__(answers)
        self.calls = 0
        self.max_calls = max_calls

    def _format_answer_fields(self, field_names_and_values):
        self.calls += 1
        if self.calls > self.max_calls:
            raise AssertionError(f"LM was called {self.calls} times, expected at most {self.max_calls}")
        return super()._format_answer_fields(field_names_and_values)


def _make_avatar(max_iters: int) -> Avatar:
    return Avatar(
        signature="question -> answer",
        tools=[Tool(tool=EchoTool(), name="echo", desc="echoes the query")],
        max_iters=max_iters,
    )


def test_avataroptimizer_constructs_without_typedpredictor():
    # AvatarOptimizer is exported from dspy.teleprompt and must be constructible.
    optimizer = AvatarOptimizer(metric=lambda example, pred, trace=None: 1.0)
    assert optimizer.comparator is not None


def test_avatar_finish_flow_returns_answer():
    avatar = _make_avatar(max_iters=3)
    lm = _CountingDummyLM(
        answers=[
            {"action_1": Action(tool_name="Finish", tool_input_query="")},
            {"answer": "done"},
        ],
        max_calls=2,
    )
    dspy.configure(lm=lm)
    result = avatar(question="hello")
    assert lm.calls == 2
    assert result.answer == "done"
    assert result.actions == []


def test_avatar_honors_constructor_max_iters():
    # The constructor budget limits tool turns; without it the loop consumed
    # scripted responses until the LM had nothing left to say.
    avatar = _make_avatar(max_iters=2)
    lm = _CountingDummyLM(
        answers=[
            {"action_1": Action(tool_name="echo", tool_input_query="q1")},
            {"action_2": Action(tool_name="echo", tool_input_query="q2")},
            {"action_3": Action(tool_name="echo", tool_input_query="q3"), "answer": "done"},
        ],
        max_calls=3,
    )
    dspy.configure(lm=lm)
    result = avatar(question="hello")
    assert lm.calls == 3  # two tool turns plus the final-answer call
    assert [action.tool_name for action in result.actions] == ["echo", "echo"]
    assert result.answer == "done"


def test_avatar_max_iters_forward_override_still_works():
    avatar = _make_avatar(max_iters=10)
    lm = _CountingDummyLM(
        answers=[
            {"action_1": Action(tool_name="echo", tool_input_query="q1")},
            {"action_2": Action(tool_name="echo", tool_input_query="q2"), "answer": "done"},
        ],
        max_calls=2,
    )
    dspy.configure(lm=lm)
    result = avatar(question="hello", max_iters=1)
    assert lm.calls == 2
    assert result.answer == "done"
