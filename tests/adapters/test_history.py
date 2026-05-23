from dspy.adapters.types.history import History, make_truncate_oldest_actions, truncate_oldest_actions
from dspy.adapters.types.tool import ToolCallResults, ToolCalls


def test_messages_key_is_canonical_history_shape():
    legacy_message = {"question": "What is the capital of France?", "answer": "Paris"}

    history = History(messages=[legacy_message])

    assert history.messages[0] == legacy_message
    assert history.model_dump() == {"messages": [legacy_message]}


def test_append_adds_plain_messages():
    tool_calls = ToolCalls.from_dict_list([{"name": "search", "args": {"query": "hello"}, "id": "call_0"}])
    tool_results = ToolCallResults.from_dict_list(
        [{"call_id": "call_0", "name": "search", "value": "result", "is_error": False}]
    )
    history = History(messages=[])

    history.append(
        {
            "question": "hi",
            "next_thought": "search first",
            "tool_calls": tool_calls,
            "tool_call_results": tool_results,
            "answer": "bye",
        }
    )

    assert history.messages == [
        {
            "question": "hi",
            "next_thought": "search first",
            "tool_calls": tool_calls,
            "tool_call_results": tool_results,
            "answer": "bye",
        },
    ]
    assert History.model_validate(history.model_dump()) == history


def test_history_normalizes_raw_tool_call_lists_on_canonical_keys():
    history = History(
        messages=[
            {"tool_calls": [{"name": "search", "args": {"query": "hello"}, "id": "call_0"}]},
            {"tool_call_results": [{"call_id": "call_0", "name": "search", "value": "result"}]},
        ]
    )

    assert history.messages[0]["tool_calls"] == ToolCalls.from_dict_list(
        [{"name": "search", "args": {"query": "hello"}, "id": "call_0"}]
    )
    assert history.messages[1]["tool_call_results"] == ToolCallResults.from_dict_list(
        [{"call_id": "call_0", "name": "search", "value": "result"}]
    )


def test_compact_if_needed_calls_compact_fn_with_history():
    calls = []
    history = History(messages=[], compact_fn=calls.append)

    history.compact_if_needed()

    assert calls == [history]


def test_truncate_oldest_actions_keeps_recent_complete_tool_turns_and_non_actions():
    history = History(messages=[{"question": "new"}, {"question": "legacy", "answer": "legacy answer"}])

    for index in range(5):
        history.append(
            {
                "next_thought": str(index),
                "tool_calls": ToolCalls.from_dict_list(
                    [{"name": "search", "args": {"query": str(index)}, "id": f"call_{index}"}]
                ),
                "tool_call_results": ToolCallResults.from_dict_list(
                    [{"call_id": f"call_{index}", "name": "search", "value": f"result {index}"}]
                ),
            }
        )
    history.append({"answer": "done"})

    truncate_oldest_actions(history, max_tokens=0, keep_n=2)

    action_messages = [message for message in history.messages if "tool_calls" in message]
    result_messages = [message for message in history.messages if "tool_call_results" in message]
    assert [message["next_thought"] for message in action_messages] == ["3", "4"]
    assert [message["tool_call_results"].tool_call_results[0].value for message in result_messages] == [
        "result 3",
        "result 4",
    ]
    assert history.messages[0] == {"question": "new"}
    assert history.messages[1] == {"question": "legacy", "answer": "legacy answer"}
    assert history.messages[-1] == {"answer": "done"}


def test_make_truncate_oldest_actions_returns_compaction_fn():
    history = History(messages=[])
    for index in range(4):
        history.append(
            {
                "next_thought": str(index),
                "tool_calls": ToolCalls.from_dict_list([{"name": "search", "args": {}, "id": f"call_{index}"}]),
                "tool_call_results": ToolCallResults.from_dict_list(
                    [{"call_id": f"call_{index}", "name": "search", "value": f"result {index}"}]
                ),
            }
        )

    make_truncate_oldest_actions(max_tokens=0, keep_n=1)(history)

    action_messages = [message for message in history.messages if "tool_calls" in message]
    assert [message["next_thought"] for message in action_messages] == ["3"]
