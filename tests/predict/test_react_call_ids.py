import asyncio

import pytest

import dspy


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("storage", ["object", "dict", "json"])
@pytest.mark.parametrize("forced", [False, True])
def test_continued_react_generates_unique_call_ids(asynchronous, storage, forced):
    def lookup(query: str) -> str:
        return f"found {query}"

    replies = []
    for query in ("cats", "dogs"):
        replies.extend(
            [
                {
                    "next_thought": "Look up the query.",
                    "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "lookup", "args": {"query": query}}]),
                },
                {
                    "next_thought": "Done.",
                    "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": query}}]),
                },
            ]
        )
    agent = dspy.ReAct("question -> answer", tools=[lookup])
    lm = dspy.utils.DummyLM(replies)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        first = asyncio.run(agent.acall(question="cats")) if asynchronous else agent(question="cats")
        assert [event["tool_calls"].tool_calls[0].id for event in first.history.messages] == ["call_0_0", "call_1_0"]
        history = first.history
        if storage == "dict":
            history = history.model_dump(mode="json")
        elif storage == "json":
            history = dspy.History.model_validate_json(history.model_dump_json())
        kwargs = {"question": "dogs", "history": history, "max_iters": 1 if forced else 20}
        second = asyncio.run(agent.acall(**kwargs)) if asynchronous else agent(**kwargs)

    events = [dspy.ToolCalls.model_validate(event["tool_calls"]) for event in second.history.messages]
    calls = [call for event in events for call in event.tool_calls]
    assert second.answer == "dogs"
    assert second.termination_reason == ("forced_submit" if forced else "submit")
    assert len({call.id for call in calls}) == len(calls)
    for tool_calls in events:
        results = dspy.ToolCallResults.model_validate(tool_calls.tool_call_results)
        assert [result.call_id for result in results.tool_call_results] == [call.id for call in tool_calls.tool_calls]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_generated_ids_do_not_shadow_ids_supplied_later_in_the_batch(asynchronous):
    def lookup() -> str:
        return "evidence"

    continuation = [{"provider": "gemini", "kind": "thought_signature", "data": {"value": "opaque"}}]
    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "Look up evidence.",
                "tool_calls": {
                    "tool_calls": [
                        {"name": "lookup", "args": {}},
                        {"id": "call_0_0", "name": "lookup", "args": {}, "continuation": continuation},
                        {"id": "call_0_0_1", "name": "lookup", "args": {}},
                    ]
                },
            },
            {
                "next_thought": "Done.",
                "tool_calls": {
                    "tool_calls": [
                        {"id": "provided-submit", "name": "submit", "args": {"answer": "done"}},
                    ]
                },
            },
        ]
    )
    agent = dspy.ReAct("question -> answer", tools=[lookup])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = asyncio.run(agent.acall(question="evidence")) if asynchronous else agent(question="evidence")
    calls = result.history.messages[0]["tool_calls"].tool_calls
    assert [call.id for call in calls] == ["call_0_0_2", "call_0_0", "call_0_0_1"]
    assert calls[1].continuation == continuation
    assert result.history.messages[-1]["tool_calls"].tool_calls[0].id == "provided-submit"
