import warnings

import pytest

import dspy
from dspy.predict.react_v2 import ReActV2


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_react_v2_compatibility_name_warns_and_saved_state_loads_into_react(tmp_path, use_async):
    with pytest.warns(
        DeprecationWarning,
        match=r"dspy\.ReActV2 implementation has now replaced dspy\.ReAct.*migrate directly.*removed in DSPy 3\.6",
    ) as caught:
        legacy = ReActV2("question -> answer: int", tools=[], max_iters=0)
    assert caught[0].filename == __file__
    assert dspy.ReActV2 is ReActV2
    assert isinstance(legacy, dspy.ReAct)

    legacy.react.signature = legacy.react.signature.with_instructions("Submit the integer answer.")
    path = tmp_path / "agent.json"
    legacy.save(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        canonical = dspy.ReAct("question -> answer: int", tools=[], max_iters=0)
        canonical.load(path)
    assert canonical.react.signature.instructions == "Submit the integer answer."

    for agent in (legacy, canonical):
        lm = dspy.utils.DummyLM([
            {
                "next_thought": "Done.",
                "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": 42}}]),
            },
        ])
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
            result = await agent.acall(question="6 times 7?") if use_async else agent(question="6 times 7?")
        assert result.answer == 42
        assert result.termination_reason == "forced_submit"
        assert result.history.messages[0]["question"] == "6 times 7?"
