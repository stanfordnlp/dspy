import asyncio

import pytest
from pydantic import BaseModel

import dspy
from dspy.utils.exceptions import ContextWindowExceededError
from tests.test_utils.engines import make_response, recording_lm


class ReasoningDummyLM(dspy.utils.DummyLM):
    @property
    def supports_reasoning(self):
        return True


def test_react_submit_tool_returns_original_output_fields():
    react = dspy.ReAct("question -> answer", tools=[])

    assert react.tools["submit"](answer="Paris") == {"answer": "Paris"}
    assert "tool_call_results" not in react.react.signature.input_fields


@pytest.mark.parametrize("native", [False, True])
def test_react_preserves_output_field_metadata_in_lm_request(native):
    class Task(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(desc="Return the answer in Spanish.")
        confidence: int = dspy.OutputField(desc="An integer confidence percentage from 0 to 100.", ge=0, le=100)

    outputs = {"answer": "París", "confidence": 95}
    if native:
        lm = recording_lm([make_response(tool_calls=[("submit_1", "submit", outputs)])])
    else:
        lm = dspy.utils.DummyLM([{
            "next_thought": "Done.",
            "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": outputs}]),
        }])

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=native)):
        result = dspy.ReAct(Task, tools=[])(question="What is the capital of France?")

    assert result.answer == "París"
    assert result.confidence == 95
    if native:
        submit = next(tool for tool in lm.engine.requests[0].tools if tool.name == "submit")
        assert submit.parameters["properties"] == {
            "answer": {"type": "string", "description": "Return the answer in Spanish."},
            "confidence": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "An integer confidence percentage from 0 to 100.",
            },
        }
    else:
        prompt = str(lm.history[0]["messages"])
        assert "Return the answer in Spanish." in prompt
        assert "An integer confidence percentage from 0 to 100." in prompt


def test_react_text_mock_lm_loop_records_inputs_once():
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "I should look this up.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [{"name": "lookup", "args": {"query": "cats"}}]
                ),
            },
            {
                "next_thought": "I can answer now.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [{"name": "submit", "args": {"answer": "found cats"}}]
                ),
            },
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = dspy.ReAct("question -> answer", tools=[lookup])(question="cats")

    assert pred.answer == "found cats"
    assert pred.termination_reason == "submit"
    assert sum("question" in event for event in pred.history.messages) == 1
    assert pred.history.messages[0]["tool_calls"].tool_calls[0].id == "call_0_0"
    assert "tool_call_results" not in pred.history.messages[0]
    assert pred.history.messages[0]["tool_calls"].tool_call_results.tool_call_results[0].call_id == "call_0_0"


def test_react_continuation_omits_missing_original_inputs():
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "I should look this up.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [{"name": "lookup", "args": {"query": "cats"}}]
                ),
            },
            {
                "next_thought": "I can answer now.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [{"name": "submit", "args": {"answer": "found cats"}}]
                ),
            },
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = dspy.ReAct("question -> answer", tools=[lookup])(question="cats")

    assert pred.answer == "found cats"
    second_call_messages = lm.history[1]["messages"]
    second_current_user_message = second_call_messages[-1]["content"]
    assert "[[ ## question ## ]]\nNone" not in second_current_user_message
    assert "[[ ## question ## ]]" not in second_current_user_message
    assert any("[[ ## question ## ]]\ncats" in message["content"] for message in second_call_messages)


def test_react_text_mode_accepts_top_level_tool_arguments():
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
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [{"name": "submit", "args": {"answer": "found cats"}}]
                ),
            },
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=False)):
        pred = dspy.ReAct("question -> answer", tools=[lookup])(question="cats")

    assert pred.answer == "found cats"
    assert pred.termination_reason == "submit"
    assert pred.history.messages[0]["tool_calls"].tool_calls[0].args == {"query": "cats"}


def test_react_text_mode_accepts_wrapped_submit_arguments():
    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "I can answer now.",
                "tool_calls": {"tool_calls": [{"name": "submit", "arguments": {"answer": "done"}}]},
            },
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=False)):
        pred = dspy.ReAct("question -> answer", tools=[])(question="cats")

    assert pred.answer == "done"
    assert pred.termination_reason == "submit"


def test_react_unknown_tool_observation_can_continue():
    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "Try a missing tool.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [{"name": "missing_tool", "args": {"query": "cats"}}]
                ),
            },
            {
                "next_thought": "Recover with a final answer.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [{"name": "submit", "args": {"answer": "done"}}]
                ),
            },
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = dspy.ReAct("question -> answer", tools=[])(question="cats")

    first_result = pred.history.messages[0]["tool_calls"].tool_call_results.tool_call_results[0]
    assert first_result.is_error is True
    assert first_result.call_id == "call_0_0"
    assert "Unknown tool" in first_result.value
    assert pred.answer == "done"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("adapter_type", [dspy.ChatAdapter, dspy.JSONAdapter, dspy.XMLAdapter])
def test_react_native_replay_preserves_tool_error_status(asynchronous, adapter_type):
    def lookup(number: int) -> int:
        if number < 0:
            raise ValueError("negative-input-73")
        return number * 3 + 1

    lm = recording_lm([
        make_response(tool_calls=[
            ("missing-73", "missing_lookup", {}),
            ("success-29", "lookup", {"number": 7}),
            ("failure-91", "lookup", {"number": -1}),
        ]),
        make_response(tool_calls=[("submit-37", "submit", {"answer": "done"})]),
    ])
    agent = dspy.ReAct("question -> answer", tools=[lookup])
    with dspy.context(lm=lm, adapter=adapter_type(use_native_function_calling=True)):
        result = asyncio.run(agent.acall(question="look up values")) if asynchronous else agent(question="look up values")

    assert result.answer == "done"
    replayed = [part for message in lm.engine.requests[1].messages if message.role == "tool" for part in message.parts]
    assert [(part.id, part.is_error) for part in replayed] == [
        ("missing-73", True), ("success-29", False), ("failure-91", True),
    ]
    assert "Unknown tool: missing_lookup" in replayed[0].content[0].text
    assert replayed[1].content[0].text == "22"
    assert "negative-input-73" in replayed[2].content[0].text


def test_react_accepts_serialized_history_input():
    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "I can answer.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [{"name": "submit", "args": {"answer": "done"}}]
                ),
            }
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = dspy.ReAct("question -> answer", tools=[])(history={"messages": [{"question": "old"}]})

    assert pred.answer == "done"
    assert pred.history.messages[0] == {"question": "old"}
    assert all(event for event in pred.history.messages)


def test_react_forced_submit_on_empty_tool_calls():
    lm = ReasoningDummyLM(
        [
            {"next_thought": "No action.", "tool_calls": dspy.ToolCalls(tool_calls=[])},
            {
                "next_thought": "Forced final.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [{"name": "submit", "args": {"answer": "forced"}}]
                ),
            },
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = dspy.ReAct("question -> answer", tools=[])(question="cats")

    assert pred.answer == "forced"
    assert pred.termination_reason == "forced_submit"
    assert lm.history[0]["request"].config.reasoning.effort == "low"
    assert lm.history[1]["request"].config.tool_choice is None
    assert lm.history[1]["request"].config.reasoning is None


@pytest.mark.parametrize("adapter", [None, dspy.JSONAdapter()], ids=["default", "json"])
def test_react_forced_submit_with_native_flag_but_unsupported_lm(adapter):
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
        pred = dspy.ReAct("question -> answer", tools=[])(question="cats")

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
def test_react_native_tool_loop_replays_tool_result_with_provider_id(adapter, max_iters):
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = NativeToolLM()

    with dspy.context(lm=lm, adapter=adapter):
        pred = dspy.ReAct("question -> answer", tools=[lookup], max_iters=max_iters)(question="cats")

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
def test_react_native_parallel_tool_calls_are_requested_and_replayed(adapter):
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = ParallelNativeToolLM()

    with dspy.context(lm=lm, adapter=adapter):
        pred = dspy.ReAct("question -> answer", tools=[lookup])(question="cats and dogs")

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
        result.call_id
        for result in pred.history.messages[0]["tool_calls"].tool_call_results.tool_call_results
    ] == [
        "call_provider_1",
        "call_provider_2",
    ]
    assert [
        message["tool_call_id"]
        for message in lm.calls[1]["messages"]
        if message["role"] == "tool"
    ] == [
        "call_provider_1",
        "call_provider_2",
    ]


@pytest.mark.parametrize("reserved", ["history", "termination_reason"])
def test_react_rejects_reserved_output_field_names(reserved):
    with pytest.raises(ValueError, match=reserved):
        dspy.ReAct(f"question -> answer, {reserved}: str", tools=[])


@pytest.mark.parametrize("reserved", ["history", "tools", "turn_budget"])
def test_react_rejects_reserved_input_field_names(reserved):
    with pytest.raises(ValueError, match=reserved):
        dspy.ReAct(f"{reserved}: str -> answer: str", tools=[])


def _lookup_turn(query):
    return {
        "next_thought": f"Look up {query}.",
        "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "lookup", "args": {"query": query}}]),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_react_shows_current_turn_budget_each_call(use_async):
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = dspy.utils.DummyLM(
        [
            _lookup_turn("cats"),
            _lookup_turn("dogs"),
            {
                "next_thought": "Out of turns.",
                "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "pets"}}]),
            },
        ]
    )
    react = dspy.ReAct("question -> answer", tools=[lookup], max_iters=20)

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        if use_async:
            pred = await react.acall(question="pets", max_iters=2)
        else:
            pred = react(question="pets", max_iters=2)

    assert pred.termination_reason == "forced_submit"
    assert "how many turns remain before you must submit" in lm.history[0]["messages"][0]["content"]
    expected = [
        "Turn 1 of 2 (1 more after this one).",
        "Turn 2 of 2 (this is your last turn).",
        "No turns remain. Call `submit` now with your final outputs.",
    ]
    for call, budget in zip(lm.history, expected, strict=True):
        assert f"[[ ## turn_budget ## ]]\n{budget}" in call["messages"][-1]["content"]
    # Each turn's budget is stored with that turn so replay rebuilds the exact request.
    assert [event["turn_budget"] for event in pred.history.messages] == expected


def test_react_native_requests_are_prefixes_for_prompt_caching():
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = NativeToolLM()
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=True)):
        pred = dspy.ReAct("question -> answer", tools=[lookup])(question="cats")

    assert pred.answer == "found cats"
    first, second = (call["messages"] for call in lm.calls)
    assert second[: len(first)] == first
    assert "[[ ## turn_budget ## ]]\nTurn 2 of 20" in second[-1]["content"]


def test_react_agent_pickled_before_turn_budget_keeps_old_behavior():
    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "Done.",
                "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "done"}}]),
            }
        ]
    )
    react = dspy.ReAct("question -> answer", tools=[])
    # An agent pickled before this field existed restores its old inner signature without `__init__`.
    react.react.signature = react.react.signature.delete("turn_budget")

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = react(question="cats")

    assert pred.answer == "done"
    assert all("turn_budget" not in message["content"] for message in lm.history[0]["messages"][1:])
    assert all("turn_budget" not in event for event in pred.history.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("break_reason", ["max_iters", "empty_tool_calls", "parse_error", "context_window_exceeded"])
@pytest.mark.parametrize("forced_result", ["no_submit", "missing_field", "parse_error", "context_window_exceeded"])
async def test_react_failed_termination_raises(mocker, break_reason, forced_result, use_async):
    react = dspy.ReAct("question -> answer: str, count: int", tools=[], max_iters=1)
    empty = dspy.Prediction(tool_calls=dspy.ToolCalls(tool_calls=[]))
    incomplete = dspy.Prediction(
        tool_calls=dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "partial"}}])
    )
    outcomes = {
        "max_iters": incomplete,
        "empty_tool_calls": empty,
        "no_submit": empty,
        "missing_field": incomplete,
        "parse_error": ValueError("invalid tool calls"),
        "context_window_exceeded": ContextWindowExceededError(message="too long"),
    }
    mocker.patch.object(
        react.react,
        "aforward" if use_async else "forward",
        side_effect=[outcomes[break_reason], outcomes[forced_result]],
    )

    history = dspy.History(messages=[])
    error_type = ContextWindowExceededError if forced_result == "context_window_exceeded" else ValueError
    with pytest.raises(error_type) as exc:
        if use_async:
            await react.acall(question="cats", history=history)
        else:
            react(question="cats", history=history)

    if forced_result in {"parse_error", "context_window_exceeded"}:
        assert exc.value is outcomes[forced_result]
    else:
        assert f"failed to produce final outputs after {break_reason}" in str(exc.value)
    if forced_result == "no_submit":
        assert "no submit call" in str(exc.value)
    if forced_result == "missing_field":
        assert "Missing required final output field(s): count" in str(exc.value)
        event = history.messages[-1]
        assert event["tool_calls"].tool_calls[0].args == {"answer": "partial"}
        assert event["tool_calls"].tool_call_results.tool_call_results[0].is_error is True


def test_react_evaluate_reports_submission_failure(caplog):
    lm = dspy.utils.DummyLM([{"next_thought": "No answer.", "tool_calls": dspy.ToolCalls(tool_calls=[])}] * 2)

    def metric(example, pred, trace=None):
        pytest.fail("A failed submission must not reach the metric")

    evaluate = dspy.Evaluate(
        devset=[dspy.Example(question="cats", answer="felines").with_inputs("question")],
        metric=metric,
        num_threads=1,
        max_errors=1,
        display_progress=False,
    )
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        with pytest.raises(Exception, match="Execution cancelled due to errors"):
            evaluate(dspy.ReAct("question -> answer", tools=[]))
    assert "failed to produce final outputs after empty_tool_calls: no submit call" in caplog.text


@pytest.mark.asyncio
async def test_react_async_mixed_tools_run_sequentially_and_recover():
    executed = []

    async def lookup(query: str) -> str:
        await asyncio.sleep(0)
        executed.append(query)
        return f"found {query}"

    def fail() -> str:
        executed.append("fail")
        raise ValueError("lookup unavailable")

    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "Look up both.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [
                        {"name": "lookup", "args": {"query": "cats"}},
                        {"name": "missing", "args": {}},
                        {"name": "fail", "args": {}},
                        {"name": "lookup", "args": {"query": "dogs"}},
                    ]
                ),
            },
            {
                "next_thought": "Done.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [
                        {"name": "submit", "args": {"answer": "cats and dogs", "count": 2}},
                    ]
                ),
            },
        ]
    )
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = await dspy.ReAct("question -> answer: str, count: int", tools=[lookup, fail]).acall(
            question="pets",
            history={"messages": [{"question": "old"}]},
        )

    assert (pred.answer, pred.count, pred.termination_reason) == ("cats and dogs", 2, "submit")
    assert executed == ["cats", "fail", "dogs"]
    assert pred.history.messages[0] == {"question": "old"}
    assert pred.history.messages[1]["question"] == "pets"
    assert "question" not in pred.history.messages[2]
    results = pred.history.messages[1]["tool_calls"].tool_call_results.tool_call_results
    assert [r.call_id for r in results] == ["call_0_0", "call_0_1", "call_0_2", "call_0_3"]
    assert [r.is_error for r in results] == [False, True, True, False]
    assert results[0].value == "found cats"
    assert results[1].value == "Unknown tool: missing"
    assert "lookup unavailable" in results[2].value
    assert results[3].value == "found dogs"


@pytest.mark.asyncio
@pytest.mark.parametrize("max_iters", [0, 1])
async def test_react_async_forced_submit_filters_other_tools(mocker, max_iters):
    def forbidden() -> str:
        pytest.fail("Forced submit must not execute other tools")

    react = dspy.ReAct("question -> answer", tools=[forbidden], max_iters=20)
    empty = dspy.Prediction(tool_calls=dspy.ToolCalls(tool_calls=[]))
    final = dspy.Prediction(
        tool_calls=dspy.ToolCalls.from_dict_list(
            [
                {"name": "forbidden", "args": {}},
                {"name": "submit", "args": {"answer": "forced"}},
            ]
        )
    )
    predictor = mocker.patch.object(react.react, "aforward", side_effect=[empty] * max_iters + [final])

    pred = await react.acall(question="cats", max_iters=max_iters)

    assert pred.answer == "forced"
    assert pred.termination_reason == "forced_submit"
    assert pred.history.messages[0]["question"] == "cats"
    assert pred.history.messages[0]["tool_calls"].tool_calls[0].id == f"call_{max_iters}_1"
    assert predictor.call_args.kwargs["config"] == {
        "tool_choice": {"type": "function", "function": {"name": "submit"}},
        "reasoning_effort": None,
    }


@pytest.mark.asyncio
async def test_react_async_tool_cancellation_propagates(mocker):
    async def cancel() -> str:
        raise asyncio.CancelledError

    react = dspy.ReAct("question -> answer", tools=[cancel])
    predictor = mocker.patch.object(
        react.react,
        "aforward",
        return_value=dspy.Prediction(
            tool_calls=dspy.ToolCalls.from_dict_list([{"name": "cancel", "args": {}}]),
        ),
    )
    with pytest.raises(asyncio.CancelledError):
        await react.acall(question="cats")
    assert predictor.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_tool_calling_with_pydantic_args(use_async):
    class CalendarEvent(BaseModel):
        name: str
        participants: list[str]

    def invite(event: CalendarEvent) -> str:
        return f"{', '.join(event.participants)}: {event.name}"

    async def async_invite(event: CalendarEvent) -> str:
        return invite(event)

    tool = dspy.Tool(async_invite if use_async else invite, name="invite")
    lm = dspy.utils.DummyLM([
        {
            "next_thought": "Invite the participants.",
            "tool_calls": dspy.ToolCalls.from_dict_list([
                {"name": "invite", "args": {"event": {"name": "Science Fair", "participants": ["Alice", "Bob"]}}},
            ]),
        },
        {
            "next_thought": "Done.",
            "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "invited"}}]),
        },
    ])
    agent = dspy.ReAct("question -> answer", tools=[tool])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = await agent.acall(question="Invite everyone") if use_async else agent(question="Invite everyone")
    observation = result.history.messages[0]["tool_calls"].tool_call_results.tool_call_results[0]
    assert observation.is_error is False
    assert observation.value == "Alice, Bob: Science Fair"
    assert result.answer == "invited"


@pytest.mark.extra
def test_tool_observation_preserves_custom_type():
    image = pytest.importorskip("PIL.Image")

    def make_images():
        return dspy.Image("https://example.com/test.png"), dspy.Image(image.new("RGB", (100, 100), "red"))

    lm = dspy.utils.DummyLM([
        {
            "next_thought": "Get the images.",
            "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "make_images", "args": {}}]),
        },
        {
            "next_thought": "Done.",
            "tool_calls": dspy.ToolCalls.from_dict_list([{"name": "submit", "args": {"answer": "done"}}]),
        },
    ])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = dspy.ReAct("question -> answer", tools=[make_images])(question="Draw something red")

    images = [
        part
        for message in lm.history[1]["messages"]
        if isinstance(message["content"], list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image"
    ]
    assert len(images) == 2
    assert result.answer == "done"
