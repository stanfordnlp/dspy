import warnings

import pytest

import dspy
from dspy.predict.react_v2 import ReActV2
from tests.test_utils.engines import make_response


class ReasoningDummyLM(dspy.utils.DummyLM):
    @property
    def supports_reasoning(self):
        return True


def test_react_v2_submit_tool_returns_original_output_fields():
    react = dspy.ReActV2("question -> answer", tools=[])

    assert react.tools["submit"](answer="Paris") == {"answer": "Paris"}
    assert "tool_call_results" not in react.react.signature.input_fields


def test_react_v2_text_mock_lm_loop_records_inputs_once():
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "I should look this up.",
                "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "lookup", "args": {"query": "cats"}}]),
            },
            {
                "next_thought": "I can answer now.",
                "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "found cats"}}]),
            },
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = dspy.ReActV2("question -> answer", tools=[lookup])(question="cats")

    assert pred.answer == "found cats"
    assert pred.termination_reason == "submit"
    assert sum("question" in event for event in pred.history.messages) == 1
    assert pred.history.messages[0]["tool_calls"].tool_calls[0].id == "call_0_0"
    assert "tool_call_results" not in pred.history.messages[0]
    assert pred.history.messages[0]["tool_calls"].tool_call_results.tool_call_results[0].call_id == "call_0_0"


def test_react_v2_continuation_omits_missing_original_inputs():
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "I should look this up.",
                "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "lookup", "args": {"query": "cats"}}]),
            },
            {
                "next_thought": "I can answer now.",
                "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "found cats"}}]),
            },
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = dspy.ReActV2("question -> answer", tools=[lookup])(question="cats")

    assert pred.answer == "found cats"
    second_call_messages = lm.history[1]["messages"]
    second_current_user_message = second_call_messages[-1]["content"]
    assert "[[ ## question ## ]]\nNone" not in second_current_user_message
    assert "[[ ## question ## ]]" not in second_current_user_message
    assert any("[[ ## question ## ]]\ncats" in message["content"] for message in second_call_messages)


def test_react_v2_text_mode_accepts_top_level_tool_arguments():
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "I should look this up.",
                "tool_calls": {"name": "lookup", "arguments": {"query": "cats"}},
            },
            {
                "next_thought": "I can answer now.",
                "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "found cats"}}]),
            },
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=False)):
        pred = dspy.ReActV2("question -> answer", tools=[lookup])(question="cats")

    assert pred.answer == "found cats"
    assert pred.termination_reason == "submit"
    assert pred.history.messages[0]["tool_calls"].tool_calls[0].args == {"query": "cats"}


def test_react_v2_text_mode_accepts_wrapped_submit_arguments():
    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "I can answer now.",
                "tool_calls": {"tool_calls": [{"name": "submit", "arguments": {"answer": "done"}}]},
            },
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=False)):
        pred = dspy.ReActV2("question -> answer", tools=[])(question="cats")

    assert pred.answer == "done"
    assert pred.termination_reason == "submit"


def test_react_v2_unknown_tool_observation_can_continue():
    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "Try a missing tool.",
                "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "missing_tool", "args": {"query": "cats"}}]),
            },
            {
                "next_thought": "Recover with a final answer.",
                "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "done"}}]),
            },
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = dspy.ReActV2("question -> answer", tools=[])(question="cats")

    first_result = pred.history.messages[0]["tool_calls"].tool_call_results.tool_call_results[0]
    assert first_result.is_error is True
    assert first_result.call_id == "call_0_0"
    assert "Unknown tool" in first_result.value
    assert pred.answer == "done"


def test_react_v2_accepts_serialized_history_input():
    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "I can answer.",
                "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "done"}}]),
            }
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = dspy.ReActV2("question -> answer", tools=[])(history={"messages": [{"question": "old"}]})

    assert pred.answer == "done"
    assert pred.history.messages[0] == {"question": "old"}
    assert all(event for event in pred.history.messages)
    assert len(pred.history.messages) == 2


def test_react_v2_forced_submit_on_empty_tool_calls():
    lm = ReasoningDummyLM(
        [
            {"next_thought": "No action.", "tool_calls": dspy.ToolCalls(tool_calls=[])},
            {
                "next_thought": "Forced final.",
                "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "forced"}}]),
            },
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = dspy.ReActV2("question -> answer", tools=[])(question="cats")

    assert pred.answer == "forced"
    assert pred.termination_reason == "forced_submit"
    assert lm.history[0]["request"].config.reasoning.effort == "low"
    assert lm.history[1]["request"].config.tool_choice is None
    assert lm.history[1]["request"].config.reasoning is None


@pytest.mark.parametrize("adapter", [None, dspy.JSONAdapter()], ids=["default", "json"])
def test_react_v2_forced_submit_with_native_flag_but_unsupported_lm(adapter):
    lm = dspy.utils.DummyLM(
        [
            {"next_thought": "No action.", "tool_calls": dspy.ToolCalls(tool_calls=[])},
            {
                "next_thought": "Forced final.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [{"name": "submit", "args": {"answer": "forced"}}]
                ),
            },
        ],
        adapter=adapter,
    )
    assert lm.supports_function_calling is False

    with dspy.context(lm=lm, adapter=adapter):
        pred = dspy.ReActV2("question -> answer", tools=[])(question="cats")

    assert pred.answer == "forced"
    assert pred.termination_reason == "forced_submit"
    assert "tool_choice" not in lm.history[1]["kwargs"]
    assert "tools" not in lm.history[1]["kwargs"]


class _ToolCallEngine:
    """Answer the first request with the given tool calls, then submit."""

    def __init__(self, first_calls, final_answer):
        self.first_calls = first_calls
        self.final_answer = final_answer
        self.requests = []
        self.supports_function_calling = True

    def complete(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            calls = self.first_calls
        else:
            calls = [("call_submit", "submit", {"answer": self.final_answer})]
        return make_response(tool_calls=calls, model=request.model)


class _ToolCallLM(dspy.LM):
    @property
    def calls(self):
        from dspy.clients.lm15_boundary import request_kwargs

        return [{"messages": request_kwargs(r, "chat")["messages"], "request": r} for r in self.engine.requests]


def NativeToolLM():
    engine = _ToolCallEngine([("call_provider_1", "lookup", {"query": "cats"})], "found cats")
    return _ToolCallLM("native-tool-lm", engine=engine, cache=False)


def ParallelNativeToolLM():
    engine = _ToolCallEngine(
        [("call_provider_1", "lookup", {"query": "cats"}), ("call_provider_2", "lookup", {"query": "dogs"})],
        "found cats and found dogs",
    )
    return _ToolCallLM("parallel-native-tool-lm", engine=engine, cache=False)


@pytest.mark.parametrize(
    "adapter", [None, dspy.ChatAdapter(use_native_function_calling=True)], ids=["default", "chat"]
)
@pytest.mark.parametrize("max_iters", [1, 20])
def test_react_v2_native_tool_loop_replays_tool_result_with_provider_id(adapter, max_iters):
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = NativeToolLM()

    with dspy.context(lm=lm, adapter=adapter):
        pred = dspy.ReActV2("question -> answer", tools=[lookup], max_iters=max_iters)(question="cats")

    assert pred.answer == "found cats"
    assert {tool.name for tool in lm.engine.requests[0].tools} == {"lookup", "submit"}
    assert pred.termination_reason == ("forced_submit" if max_iters == 1 else "submit")
    if max_iters == 1:
        assert lm.engine.requests[-1].config.tool_choice.allowed == ("submit",)
    assert pred.history.messages[0]["tool_calls"].tool_calls[0].id == "call_provider_1"
    assert "tool_call_results" not in pred.history.messages[0]
    assert pred.history.messages[0]["tool_calls"].tool_call_results.tool_call_results[0].call_id == "call_provider_1"
    assert any(
        message["role"] == "tool" and message["tool_call_id"] == "call_provider_1"
        for message in lm.calls[1]["messages"]
    )


@pytest.mark.parametrize(
    "adapter",
    [None, dspy.ChatAdapter(use_native_function_calling=True, parallel_tool_calls=True)],
    ids=["default", "chat"],
)
def test_react_v2_native_parallel_tool_calls_are_requested_and_replayed(adapter):
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = ParallelNativeToolLM()

    with dspy.context(lm=lm, adapter=adapter):
        pred = dspy.ReActV2("question -> answer", tools=[lookup])(question="cats and dogs")

    assert pred.answer == "found cats and found dogs"
    if adapter is not None:
        assert lm.calls[0]["request"].config.tool_choice.parallel is True
    else:
        assert lm.calls[0]["request"].config.tool_choice is None
    assert [call.id for call in pred.history.messages[0]["tool_calls"].tool_calls] == [
        "call_provider_1",
        "call_provider_2",
    ]
    assert [
        result.call_id for result in pred.history.messages[0]["tool_calls"].tool_call_results.tool_call_results
    ] == [
        "call_provider_1",
        "call_provider_2",
    ]
    assert [message["tool_call_id"] for message in lm.calls[1]["messages"] if message["role"] == "tool"] == [
        "call_provider_1",
        "call_provider_2",
    ]


@pytest.mark.parametrize("reserved", ["history", "termination_reason"])
def test_react_v2_rejects_reserved_output_field_names(reserved):
    with pytest.raises(ValueError, match=reserved):
        dspy.ReActV2(f"question -> answer, {reserved}: str", tools=[])


@pytest.mark.parametrize("reserved", ["history", "tools", "turn_budget"])
def test_react_v2_rejects_reserved_input_field_names(reserved):
    with pytest.raises(ValueError, match=reserved):
        dspy.ReActV2(f"{reserved}: str -> answer: str", tools=[])


def _lookup_turn(query):
    return {
        "next_thought": f"Look up {query}.",
        "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "lookup", "args": {"query": query}}]),
    }


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
