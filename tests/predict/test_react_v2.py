import asyncio

import pytest

import dspy
from dspy.dsp.utils.utils import dotdict
from dspy.utils.exceptions import ContextWindowExceededError


class ReasoningDummyLM(dspy.utils.DummyLM):
    @property
    def supports_reasoning(self):
        return True


def test_react_v2_submit_tool_returns_original_output_fields():
    react = dspy.ReActV2("question -> answer", tools=[])

    assert react.tools["submit"](answer="Paris") == {"answer": "Paris"}
    assert "tool_call_results" not in react.react.signature.input_fields


def test_react_v2_submit_recovers_value_wrapped_in_its_own_marker():
    react = dspy.ReActV2("question -> answer", tools=[])

    submitted = "[[ ## answer ## ]]\nParis\n[[ ## completed ## ]]"

    assert react.tools["submit"](answer=submitted) == {"answer": "Paris"}


def test_react_v2_submit_recovers_value_from_an_indented_marker_line():
    """An indented marker must not leave its own closing bracket in the value."""
    react = dspy.ReActV2("question -> answer", tools=[])

    assert react.tools["submit"](answer="  [[ ## answer ## ]] Paris") == {"answer": "Paris"}
    assert react.tools["submit"](answer="\t[[ ## answer ## ]]\n  Paris\n[[ ## completed ## ]]") == {
        "answer": "Paris"
    }


def test_react_v2_submit_recovers_value_emitted_before_a_marker():
    react = dspy.ReActV2("question -> answer", tools=[])

    assert react.tools["submit"](answer="Paris\n\n[[ ## completed ## ]]") == {"answer": "Paris"}


def test_react_v2_submit_rejects_scaffold_with_no_value_in_it():
    """The reported failure: the marker scaffold arrives carrying planning text only."""
    react = dspy.ReActV2("question -> answer", tools=[])

    leaked = "[[ ## next_thought ## ]]\nPreparing the answer...\n[[ ## completed ## ]]"

    with pytest.raises(ValueError) as err:
        react.tools["submit"](answer=leaked)

    message = str(err.value)
    assert "next_thought" in message
    assert "answer" in message


def test_react_v2_submit_leaves_ordinary_values_untouched():
    react = dspy.ReActV2("question -> answer", tools=[])

    assert react.tools["submit"](answer="Paris") == {"answer": "Paris"}
    assert react.tools["submit"](answer="the [[ bracket ]] stays") == {"answer": "the [[ bracket ]] stays"}


def test_react_v2_submit_leaves_non_string_values_untouched():
    react = dspy.ReActV2("question -> count: int", tools=[])

    assert react.tools["submit"](count=42) == {"count": 42}


def test_react_v2_end_to_end_recovers_marker_wrapped_answer():
    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "I can answer now.",
                "tool_calls": {
                    "tool_calls": [
                        {
                            "name": "submit",
                            "arguments": {"answer": "[[ ## answer ## ]]\n42\n[[ ## completed ## ]]"},
                        }
                    ]
                },
            },
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=False)):
        pred = dspy.ReActV2("question -> answer", tools=[])(question="what is it")

    assert pred.answer == "42"
    assert pred.termination_reason == "submit"


def test_react_v2_end_to_end_retries_after_a_leaked_scaffold():
    """A scaffold with no value is reported back as a tool error, so the model can retry."""
    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "Submitting.",
                "tool_calls": {
                    "tool_calls": [
                        {
                            "name": "submit",
                            "arguments": {
                                "answer": "[[ ## next_thought ## ]]\nPreparing...\n[[ ## completed ## ]]"
                            },
                        }
                    ]
                },
            },
            {
                "next_thought": "Retrying with a plain value.",
                "tool_calls": {"tool_calls": [{"name": "submit", "arguments": {"answer": "42"}}]},
            },
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=False)):
        pred = dspy.ReActV2("question -> answer", tools=[])(question="what is it")

    assert pred.answer == "42"


def test_react_v2_text_mock_lm_loop_records_inputs_once():
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
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [{"name": "submit", "args": {"answer": "found cats"}}]
                ),
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
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [{"name": "submit", "args": {"answer": "done"}}]
                ),
            }
        ]
    )

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = dspy.ReActV2("question -> answer", tools=[])(history={"messages": [{"question": "old"}]})

    assert pred.answer == "done"
    assert pred.history.messages[0] == {"question": "old"}
    assert all(event for event in pred.history.messages)


def test_react_v2_forced_submit_on_empty_tool_calls():
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
        pred = dspy.ReActV2("question -> answer", tools=[])(question="cats")

    assert pred.answer == "forced"
    assert pred.termination_reason == "forced_submit"
    assert lm.history[0]["kwargs"]["reasoning_effort"] == "low"
    assert "tool_choice" not in lm.history[1]["kwargs"]
    assert lm.history[1]["kwargs"].get("reasoning_effort") is None


class NativeToolLM(dspy.BaseLM):
    def __init__(self):
        super().__init__("native-tool-lm", "chat", 0.0, 1000, True)
        self.calls = []

    @property
    def supports_function_calling(self):
        return True

    def forward(self, prompt=None, messages=None, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if len(self.calls) == 1:
            tool_call = dotdict(
                id="call_provider_1",
                type="function",
                function=dotdict(name="lookup", arguments='{"query":"cats"}'),
            )
        else:
            tool_call = dotdict(
                id="call_submit",
                type="function",
                function=dotdict(name="submit", arguments='{"answer":"found cats"}'),
            )

        return dotdict(
            choices=[
                dotdict(
                    message=dotdict(content=None, tool_calls=[tool_call]),
                    finish_reason="tool_calls",
                )
            ],
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model="native-tool-lm",
        )


class ParallelNativeToolLM(dspy.BaseLM):
    def __init__(self):
        super().__init__("parallel-native-tool-lm", "chat", 0.0, 1000, True)
        self.calls = []

    @property
    def supports_function_calling(self):
        return True

    def forward(self, prompt=None, messages=None, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if len(self.calls) == 1:
            tool_calls = [
                dotdict(
                    id="call_provider_1",
                    type="function",
                    function=dotdict(name="lookup", arguments='{"query":"cats"}'),
                ),
                dotdict(
                    id="call_provider_2",
                    type="function",
                    function=dotdict(name="lookup", arguments='{"query":"dogs"}'),
                ),
            ]
        else:
            tool_calls = [
                dotdict(
                    id="call_submit",
                    type="function",
                    function=dotdict(name="submit", arguments='{"answer":"found cats and found dogs"}'),
                )
            ]

        return dotdict(
            choices=[
                dotdict(
                    message=dotdict(content=None, tool_calls=tool_calls),
                    finish_reason="tool_calls",
                )
            ],
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model="parallel-native-tool-lm",
        )


def test_react_v2_native_tool_loop_replays_tool_result_with_provider_id():
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = NativeToolLM()

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=True)):
        pred = dspy.ReActV2("question -> answer", tools=[lookup])(question="cats")

    assert pred.answer == "found cats"
    assert pred.history.messages[0]["tool_calls"].tool_calls[0].id == "call_provider_1"
    assert "tool_call_results" not in pred.history.messages[0]
    assert pred.history.messages[0]["tool_calls"].tool_call_results.tool_call_results[0].call_id == "call_provider_1"
    assert any(
        message["role"] == "tool" and message["tool_call_id"] == "call_provider_1"
        for message in lm.calls[1]["messages"]
    )


def test_react_v2_native_parallel_tool_calls_are_requested_and_replayed():
    def lookup(query: str) -> str:
        return f"found {query}"

    lm = ParallelNativeToolLM()

    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=True, parallel_tool_calls=True)):
        pred = dspy.ReActV2("question -> answer", tools=[lookup])(question="cats and dogs")

    assert pred.answer == "found cats and found dogs"
    assert lm.calls[0]["kwargs"]["parallel_tool_calls"] is True
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
def test_react_v2_rejects_reserved_output_field_names(reserved):
    with pytest.raises(ValueError, match=reserved):
        dspy.ReActV2(f"question -> answer, {reserved}: str", tools=[])


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("break_reason", ["max_iters", "empty_tool_calls", "parse_error", "context_window_exceeded"])
@pytest.mark.parametrize("forced_result", ["no_submit", "missing_field", "parse_error", "context_window_exceeded"])
async def test_react_v2_failed_termination_raises(mocker, break_reason, forced_result, use_async):
    react = dspy.ReActV2("question -> answer: str, count: int", tools=[], max_iters=1)
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


def test_react_v2_evaluate_reports_submission_failure(caplog):
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
            evaluate(dspy.ReActV2("question -> answer", tools=[]))
    assert "failed to produce final outputs after empty_tool_calls: no submit call" in caplog.text


@pytest.mark.asyncio
async def test_react_v2_async_mixed_tools_run_sequentially_and_recover():
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
        pred = await dspy.ReActV2("question -> answer: str, count: int", tools=[lookup, fail]).acall(
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
async def test_react_v2_async_forced_submit_filters_other_tools(mocker, max_iters):
    def forbidden() -> str:
        pytest.fail("Forced submit must not execute other tools")

    react = dspy.ReActV2("question -> answer", tools=[forbidden], max_iters=20)
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
async def test_react_v2_async_tool_cancellation_propagates(mocker):
    async def cancel() -> str:
        raise asyncio.CancelledError

    react = dspy.ReActV2("question -> answer", tools=[cancel])
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
