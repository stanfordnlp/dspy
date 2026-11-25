import pytest

import dspy


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("location", ["lm", "predictor", "call"])
@pytest.mark.parametrize("reasoning,effort", [
    ({"enabled": False}, "off"), ({"effort": "none"}, "off"),
    ({"effort": "off"}, "off"),
    ({"effort": "high", "summary": "detailed"}, "high"), ({}, "medium"),
])
def test_cot_reasoning_dictionary(native, location, reasoning, effort):
    from tests.test_utils.engines import make_response

    requests = []

    class Engine:
        supports_reasoning = native

        def complete(self, request):
            requests.append(request)
            plan = "<plan>explicit plan</plan>" if "<plan>" in request.system else ""
            return make_response(plan + "<answer>42</answer>", thinking="native summary")

    options = {"reasoning": reasoning}
    lm = dspy.LM("custom/reasoning", engine=Engine(), cache=False, **(options if location == "lm" else {}))
    cot = dspy.ChainOfThought("question -> answer", **(options if location == "predictor" else {}))
    with dspy.context(lm=lm, adapter=dspy.XMLAdapter(use_json_adapter_fallback=False)):
        result = cot(question="19 + 23", config=options if location == "call" else {})
    assert result.answer == "42"
    assert result.reasoning == ("native summary" if native and effort != "off" else "explicit plan")
    assert len(requests) == 1
    assert requests[0].config.reasoning.effort == effort
    if "summary" in reasoning:
        assert requests[0].config.reasoning.summary == "detailed"


@pytest.mark.parametrize("defaults,override,effort", [
    ({"reasoning": {"effort": "high"}}, {"reasoning_effort": "none"}, "off"),
    ({"reasoning_effort": "high"}, {"reasoning": {"enabled": False}}, "off"),
    ({"reasoning": {"enabled": False}}, {"reasoning_effort": "high"}, "high"),
    ({"reasoning_effort": "none"}, {"reasoning": {"effort": "high"}}, "high"),
])
def test_cot_reasoning_alias_override(defaults, override, effort):
    from tests.test_utils.engines import make_response, recording_lm

    text = "<plan>explicit plan</plan><answer>42</answer>" if effort == "off" else "<answer>42</answer>"
    lm = recording_lm([make_response(text, thinking="native summary")], **defaults)
    with dspy.context(lm=lm, adapter=dspy.XMLAdapter(use_json_adapter_fallback=False)):
        result = dspy.ChainOfThought("question -> answer")(question="19 + 23", config=override)
    assert result.reasoning == ("explicit plan" if effort == "off" else "native summary")
    assert lm.engine.requests[0].config.reasoning.effort == effort


def test_reasoning_basic_operations():
    reasoning = dspy.Reasoning(content="Hello World")

    # Test str conversion
    assert str(reasoning) == "Hello World"
    assert repr(reasoning) == "'Hello World'"

    # Test equality
    assert reasoning == "Hello World"
    assert reasoning == dspy.Reasoning(content="Hello World")
    assert reasoning != "hello world"
    assert reasoning != dspy.Reasoning(content="hello world")

    # Test len
    assert len(reasoning) == 11

    # Test indexing
    assert reasoning[0] == "H"
    assert reasoning[-1] == "d"
    assert reasoning[0:5] == "Hello"

    # Test in operator
    assert "World" in reasoning
    assert "xyz" not in reasoning

    # Test iteration
    chars = [c for c in reasoning]
    assert len(chars) == 11
    assert chars[0] == "H"


def test_reasoning_concatenation():
    reasoning = dspy.Reasoning(content="Hello")

    # Test + operator
    result1 = reasoning + " World"
    assert result1 == "Hello World"
    assert isinstance(result1, str)

    # Test reverse + operator
    result2 = "Prefix: " + reasoning
    assert result2 == "Prefix: Hello"
    assert isinstance(result2, str)

    # Test Reasoning + Reasoning
    reasoning2 = dspy.Reasoning(content=" World")
    result3 = reasoning + reasoning2
    assert isinstance(result3, dspy.Reasoning)
    assert result3.content == "Hello World"


def test_reasoning_string_methods():
    reasoning = dspy.Reasoning(content="  Hello World  ")

    # Test strip
    assert reasoning.strip() == "Hello World"

    # Test lower/upper
    assert reasoning.lower() == "  hello world  "
    assert reasoning.upper() == "  HELLO WORLD  "

    # Test split
    assert reasoning.strip().split() == ["Hello", "World"]
    assert reasoning.strip().split(" ") == ["Hello", "World"]

    # Test replace
    assert reasoning.replace("World", "Python") == "  Hello Python  "

    # Test startswith/endswith
    assert reasoning.strip().startswith("Hello")
    assert reasoning.strip().endswith("World")
    assert not reasoning.strip().startswith("World")

    # Test find
    assert reasoning.find("World") == 8
    assert reasoning.find("xyz") == -1

    # Test count
    assert reasoning.count("l") == 3

    # Test join
    assert reasoning.strip().join(["a", "b", "c"]) == "aHello WorldbHello Worldc"


def test_reasoning_with_chain_of_thought():
    from dspy.utils import DummyLM

    lm = DummyLM([{"reasoning": "Let me think step by step", "answer": "42"}])
    dspy.configure(lm=lm)

    cot = dspy.ChainOfThought("question -> answer", rationale_field_type=str)
    result = cot(question="What is the answer?")

    # Test that we can use string methods on result.reasoning
    assert isinstance(result.reasoning, str)
    assert result.reasoning.strip() == "Let me think step by step"
    assert result.reasoning.lower() == "let me think step by step"
    assert "step by step" in result.reasoning
    assert len(result.reasoning) == 25


def test_reasoning_error_message():
    reasoning = dspy.Reasoning(content="Hello")

    with pytest.raises(AttributeError, match="`Reasoning` object has no attribute 'nonexistent_method'"):
        reasoning.nonexistent_method


def test_native_reasoning_default_survives_provider_omission():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        reasoning: dspy.Reasoning = dspy.OutputField(default_factory=lambda: dspy.Reasoning(content="(omitted)"))

    adapter = dspy.ChatAdapter()
    # Native reasoning strips the field from the signature the LM formats against; the provider
    # then omits native reasoning entirely.
    processed = Sig.delete("reasoning")
    from tests.test_utils.engines import make_response

    responses = [make_response("[[ ## answer ## ]]\n42\n\n[[ ## completed ## ]]")]

    value = adapter._call_postprocess(processed, Sig, responses)[0]

    assert value["answer"] == "42"
    assert value["reasoning"] == dspy.Reasoning(content="(omitted)")


def _round_trip_history(history: dspy.History, storage: str) -> dspy.History | dict:
    if storage == "object":
        return history
    if storage == "python_dump":
        return dspy.History.model_validate(history.model_dump())
    if storage == "json_dict":
        return history.model_dump(mode="json")
    if storage == "json":
        return dspy.History.model_validate_json(history.model_dump_json())
    raise ValueError(f"Unknown storage mode: {storage}")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("storage", ["object", "python_dump", "json_dict", "json"])
@pytest.mark.parametrize("adapter_type,reply", [
    (dspy.ChatAdapter, "[[ ## answer ## ]]\n42\n\n[[ ## completed ## ]]"),
    (dspy.JSONAdapter, '{"answer": "42"}'),
    (dspy.XMLAdapter, "<answer>42</answer>"),
])
def test_exposed_thinking_replays_through_history_across_adapters(asynchronous, storage, adapter_type, reply):
    import asyncio

    from dspy.lm15 import TextPart, ThinkingPart
    from tests.test_utils.engines import make_response, recording_lm

    class ConvSig(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    lm = recording_lm([
        make_response(reply, thinking="Step 1: 19 + 23 = 42"),
        make_response(reply, thinking="Step 2: verified"),
    ])
    predict = dspy.Predict(ConvSig)
    with dspy.context(lm=lm, adapter=adapter_type()):
        first = (
            asyncio.run(predict.acall(question="First", history=dspy.History(messages=[])))
            if asynchronous
            else predict(question="First", history=dspy.History(messages=[]))
        )
        assert first.reasoning == "Step 1: 19 + 23 = 42"
        history = _round_trip_history(dspy.History(messages=[{"question": "First", **first}]), storage)
        second = (
            asyncio.run(predict.acall(question="Continue", history=history))
            if asynchronous
            else predict(question="Continue", history=history)
        )

    assert second.answer == "42"
    assistant_msgs = [m for m in lm.engine.requests[1].messages if m.role == "assistant"]
    assert len(assistant_msgs) == 1
    assert [p.text for p in assistant_msgs[0].parts_of(ThinkingPart)] == ["Step 1: 19 + 23 = 42"]
    text_parts = assistant_msgs[0].parts_of(TextPart)
    assert len(text_parts) == 1 and "42" in text_parts[0].text
    assert "Step 1" not in text_parts[0].text


@pytest.mark.parametrize("adapter_type,reply", [
    (dspy.ChatAdapter, "[[ ## answer ## ]]\n42\n\n[[ ## completed ## ]]"),
    (dspy.JSONAdapter, '{"answer": "42"}'),
    (dspy.XMLAdapter, "<answer>42</answer>"),
])
@pytest.mark.parametrize("model,expects_reasoning_content", [
    ("deepseek/deepseek-reasoner", True),
    ("openai/o3", False),
])
def test_exposed_thinking_chat_wire_respects_provider_replay_policy(
    monkeypatch, adapter_type, reply, model, expects_reasoning_content
):
    import json

    from tests.clients.test_lm_engine_execution import DualResponse, native_transport

    class ConvSig(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    def chat_body(content, reasoning_content=None):
        msg = {"role": "assistant", "content": content}
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        return DualResponse(status=200, body=json.dumps({
            "id": "c1", "model": model.split("/", 1)[1],
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
        }).encode())

    transport = native_transport(monkeypatch, [
        chat_body(reply, reasoning_content="Exposed chain of thought"),
        chat_body(reply),
    ])
    lm = dspy.LM(model, model_type="chat", engine="lm15", api_key="fake", cache=False, num_retries=0)
    predict = dspy.Predict(ConvSig)
    with dspy.context(lm=lm, adapter=adapter_type()):
        first = predict(question="First", history=dspy.History(messages=[]))
        history = dspy.History(messages=[{"question": "First", **first}])
        predict(question="Continue", history=history)

    wire_messages = json.loads(transport.requests[1].body)["messages"]
    assistant_row = next(m for m in wire_messages if m["role"] == "assistant")
    assert "Exposed chain of thought" not in (assistant_row.get("content") or "")
    if expects_reasoning_content:
        assert assistant_row["reasoning_content"] == "Exposed chain of thought"
    else:
        assert "reasoning_content" not in assistant_row


@pytest.mark.parametrize("summary", [[], [{"type": "summary_text", "text": "Readable summary"}]])
@pytest.mark.parametrize("storage", ["object", "json"])
@pytest.mark.parametrize("with_reasoning_field", [True, False])
def test_openai_responses_reasoning_item_replays_through_history_and_cache(
    monkeypatch, summary, storage, with_reasoning_field
):
    import json

    from tests.clients.test_lm_engine_execution import DualResponse, native_transport

    class WithReasoning(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    class WithoutReasoning(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()

    turn1 = DualResponse(status=200, body=json.dumps({
        "id": "resp_1", "model": "gpt-5", "status": "completed",
        "output": [
            {"type": "reasoning", "id": "rs_item_1", "encrypted_content": "enc_payload_1", "summary": summary},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "[[ ## answer ## ]]\n42\n\n[[ ## completed ## ]]"},
            ]},
        ],
        "usage": {"input_tokens": 8, "output_tokens": 5, "total_tokens": 13},
    }).encode())
    turn2 = DualResponse(status=200, body=json.dumps({
        "id": "resp_2", "model": "gpt-5", "status": "completed",
        "output": [{"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "[[ ## answer ## ]]\nConfirmed\n\n[[ ## completed ## ]]"},
        ]}],
        "usage": {"input_tokens": 16, "output_tokens": 5, "total_tokens": 21},
    }).encode())

    transport = native_transport(monkeypatch, [turn1, turn2, turn2])
    lm = dspy.LM("openai/gpt-5", model_type="responses", engine="lm15", api_key="fake", cache=True, num_retries=0)
    predict = dspy.Predict(WithReasoning if with_reasoning_field else WithoutReasoning)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        first = predict(question="First-10675", history=dspy.History(messages=[]))
        cached_first = predict(question="First-10675", history=dspy.History(messages=[]))
        assert len(transport.requests) == 1
        if with_reasoning_field:
            expected_text = "Readable summary" if summary else ""
            assert first.reasoning == expected_text
            assert cached_first.reasoning == expected_text
        history = _round_trip_history(dspy.History(messages=[{"question": "First-10675", **cached_first}]), storage)
        second = predict(question="Continue-10675", history=history, config={"cache": False})

    assert second.answer == "Confirmed"
    wire2 = json.loads(transport.requests[1].body)
    reasoning_items = [item for item in wire2["input"] if item.get("type") == "reasoning"]
    assert reasoning_items == [{
        "type": "reasoning",
        "id": "rs_item_1",
        "encrypted_content": "enc_payload_1",
        "summary": summary,
    }]


@pytest.mark.parametrize("blocks,expected_text", [
    (
        [
            {"type": "thinking", "thinking": "Signed block one. ", "signature": "sig_1"},
            {"type": "redacted_thinking", "data": "opaque_redacted_1"},
            {"type": "thinking", "thinking": "Signed block two.", "signature": "sig_2"},
        ],
        "Signed block one. Signed block two.",
    ),
    (
        [{"type": "redacted_thinking", "data": "opaque_redacted_only"}],
        "",
    ),
])
@pytest.mark.parametrize("storage", ["object", "json"])
def test_anthropic_signed_and_redacted_thinking_replays_through_history(
    monkeypatch, blocks, expected_text, storage
):
    import json

    from tests.clients.test_lm_engine_execution import DualResponse, native_transport

    class ConvSig(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    turn1 = DualResponse(status=200, body=json.dumps({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
        "content": [*blocks, {"type": "text", "text": "[[ ## answer ## ]]\n42\n\n[[ ## completed ## ]]"}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 6},
    }).encode())
    turn2 = DualResponse(status=200, body=json.dumps({
        "id": "msg_2", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "[[ ## answer ## ]]\nConfirmed\n\n[[ ## completed ## ]]"}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 18, "output_tokens": 5},
    }).encode())

    transport = native_transport(monkeypatch, [turn1, turn2])
    lm = dspy.LM("anthropic/claude-sonnet-4-5", engine="lm15", api_key="fake", cache=False, num_retries=0)
    predict = dspy.Predict(ConvSig)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        first = predict(question="First", history=dspy.History(messages=[]))
        assert first.reasoning == expected_text
        history = _round_trip_history(dspy.History(messages=[{"question": "First", **first}]), storage)
        second = predict(question="Continue", history=history)

    assert second.answer == "Confirmed"
    wire2 = json.loads(transport.requests[1].body)
    assistant_content = wire2["messages"][1]["content"]
    assert assistant_content[: len(blocks)] == blocks
    assert assistant_content[len(blocks)]["type"] == "text"


@pytest.mark.parametrize("storage", ["object", "json"])
@pytest.mark.parametrize("thought_text", ["", "Gemini thought step"])
def test_gemini_signatures_replay_on_thinking_text_and_tool_call_parts(monkeypatch, storage, thought_text):
    import json

    from tests.clients.test_lm_engine_execution import DualResponse, native_transport

    def lookup(query: str) -> str:
        return f"result:{query}"

    class ToolConvSig(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        note: str | None = dspy.OutputField(default=None)
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    turn1 = DualResponse(status=200, body=json.dumps({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"thought": True, "text": thought_text, "thoughtSignature": "sig_thought_1"},
                    {"text": "[[ ## note ## ]]\nchecking\n\n[[ ## completed ## ]]", "thoughtSignature": "sig_text_1"},
                    {"functionCall": {"id": "call_1", "name": "lookup", "args": {"query": "dspy"}},
                     "thoughtSignature": "sig_tool_1"},
                ],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 6, "totalTokenCount": 14},
    }).encode())
    turn2 = DualResponse(status=200, body=json.dumps({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"text": "[[ ## note ## ]]\ndone\n\n[[ ## completed ## ]]", "thoughtSignature": "sig_text_2"},
                    {"functionCall": {"id": "call_2", "name": "lookup", "args": {"query": "done"}}},
                ],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 18, "candidatesTokenCount": 5, "totalTokenCount": 23},
    }).encode())

    transport = native_transport(monkeypatch, [turn1, turn2])
    lm = dspy.LM("gemini/gemini-2.5-pro", engine="lm15", api_key="fake", cache=False, num_retries=0)
    predict = dspy.Predict(ToolConvSig)
    tools = [dspy.Tool(lookup)]
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter(use_native_function_calling=True)):
        first = predict(question="First", history=dspy.History(messages=[]), tools=tools)
        assert first.reasoning == thought_text
        results = dspy.ToolCallResults.from_tool_calls_and_values(first.tool_calls, ["result:dspy"])
        recorded = {
            "question": "First",
            **first,
            "tool_calls": first.tool_calls.model_copy(update={"tool_call_results": results}),
        }
        history = _round_trip_history(dspy.History(messages=[recorded]), storage)
        predict(question="Continue", history=history, tools=tools)

    wire2 = json.loads(transport.requests[1].body)
    model_parts = wire2["contents"][1]["parts"]
    assert model_parts == [
        {"text": thought_text, "thought": True, "thoughtSignature": "sig_thought_1"},
        {"text": "[[ ## note ## ]]\nchecking\n\n[[ ## completed ## ]]\n", "thoughtSignature": "sig_text_1"},
        {"functionCall": {"name": "lookup", "args": {"query": "dspy"}, "id": "call_1"},
         "thoughtSignature": "sig_tool_1"},
    ]


@pytest.mark.parametrize("target_provider", ["anthropic", "gemini", "deepseek", "textual_fallback"])
def test_incompatible_provider_or_textual_fallback_drops_foreign_opaque_state_safely(monkeypatch, target_provider):
    import json

    from dspy.lm15 import ThinkingPart
    from tests.clients.test_lm_engine_execution import DualResponse, native_transport
    from tests.test_utils.engines import make_response, recording_lm

    turn1_openai = DualResponse(status=200, body=json.dumps({
        "id": "resp_1", "model": "gpt-5", "status": "completed",
        "output": [
            {"type": "reasoning", "id": "rs_foreign", "encrypted_content": "enc_secret", "summary": []},
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "<answer>42</answer>"},
            ]},
        ],
        "usage": {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
    }).encode())

    cot = dspy.ChainOfThought("question, history: dspy.History -> answer")
    transport1 = native_transport(monkeypatch, [turn1_openai])
    openai_lm = dspy.LM("openai/gpt-5", model_type="responses", engine="lm15", api_key="fake", cache=False, num_retries=0)
    with dspy.context(lm=openai_lm, adapter=dspy.XMLAdapter(use_json_adapter_fallback=False)):
        first = cot(question="First", history=dspy.History(messages=[]))
    assert len(transport1.requests) == 1
    assert first.reasoning == ""
    history = dspy.History.model_validate_json(
        dspy.History(messages=[{"question": "First", **first}]).model_dump_json()
    )

    if target_provider == "textual_fallback":
        fallback_lm = recording_lm([make_response("<plan>explicit plan</plan><answer>84</answer>")])
        with dspy.context(lm=fallback_lm, adapter=dspy.XMLAdapter(use_json_adapter_fallback=False)):
            second = cot(question="Continue", history=history, config={"reasoning_effort": "none"})
        assert second.answer == "84" and second.reasoning == "explicit plan"
        req = fallback_lm.engine.requests[0]
        assistant_msg = next(m for m in req.messages if m.role == "assistant")
        assert assistant_msg.parts_of(ThinkingPart) == []
        assert "<plan>\n\n</plan>" in assistant_msg.text
        assert "enc_secret" not in str(req) and "__dspy_reasoning__" not in str(req)
        return

    replies = {
        "anthropic": ("anthropic/claude-sonnet-4-5", {
            "id": "m2", "type": "message", "role": "assistant", "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "<answer>84</answer>"}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 4},
        }),
        "gemini": ("gemini/gemini-2.5-pro", {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "<answer>84</answer>"}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4, "totalTokenCount": 14},
        }),
        "deepseek": ("deepseek/deepseek-reasoner", {
            "id": "c2", "model": "deepseek-reasoner",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "<answer>84</answer>"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }),
    }
    model, body = replies[target_provider]
    transport2 = native_transport(monkeypatch, [DualResponse(status=200, body=json.dumps(body).encode())])
    target_lm = dspy.LM(model, engine="lm15", api_key="fake", cache=False, num_retries=0)
    with dspy.context(lm=target_lm, adapter=dspy.XMLAdapter(use_json_adapter_fallback=False)):
        second = cot(question="Continue", history=history)
    assert second.answer == "84"
    raw_wire = transport2.requests[0].body.decode("utf-8")
    assert "enc_secret" not in raw_wire and "rs_foreign" not in raw_wire
    wire2 = json.loads(raw_wire)
    if target_provider == "anthropic":
        assert wire2["messages"][1]["content"] == [{"type": "text", "text": "<answer>\n42\n</answer>"}]
    elif target_provider == "gemini":
        assert wire2["contents"][1]["parts"] == [{"text": "<answer>\n42\n</answer>"}]
    else:
        assistant_row = next(m for m in wire2["messages"] if m["role"] == "assistant")
        assert assistant_row == {"role": "assistant", "content": "<answer>\n42\n</answer>", "reasoning_content": ""}


@pytest.mark.parametrize("storage", ["object", "json"])
def test_gemini_multi_part_text_signatures_replay_in_order(monkeypatch, storage):
    import json

    from tests.clients.test_lm_engine_execution import DualResponse, native_transport

    class ConvSig(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    turn1 = DualResponse(status=200, body=json.dumps({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {"thought": True, "text": "plan", "thoughtSignature": "sig_thought_1"},
                    {"text": "[[ ## answer ## ]]\nFirst ", "thoughtSignature": "sig_part_1"},
                    {"text": "second\n\n[[ ## completed ## ]]", "thoughtSignature": "sig_part_2"},
                ],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 6, "totalTokenCount": 14},
    }).encode())
    turn2 = DualResponse(status=200, body=json.dumps({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "[[ ## answer ## ]]\nDone\n\n[[ ## completed ## ]]"}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 16, "candidatesTokenCount": 4, "totalTokenCount": 20},
    }).encode())
    turn3 = DualResponse(status=200, body=json.dumps({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "[[ ## answer ## ]]\nEditedAck\n\n[[ ## completed ## ]]"}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 16, "candidatesTokenCount": 4, "totalTokenCount": 20},
    }).encode())

    transport = native_transport(monkeypatch, [turn1, turn2, turn3])
    lm = dspy.LM("gemini/gemini-2.5-pro", engine="lm15", api_key="fake", cache=False, num_retries=0)
    predict = dspy.Predict(ConvSig)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        first = predict(question="First", history=dspy.History(messages=[]))
        assert first.answer == "First second"

        history = _round_trip_history(dspy.History(messages=[{"question": "First", **first}]), storage)
        second = predict(question="Continue", history=history)
        assert second.answer == "Done"

        edited_history = _round_trip_history(
            dspy.History(messages=[{"question": "First", **first, "answer": "Manually edited"}]),
            storage,
        )
        third = predict(question="Verify edit", history=edited_history)
        assert third.answer == "EditedAck"

    wire2 = json.loads(transport.requests[1].body)
    assert wire2["contents"][1]["parts"] == [
        {"text": "plan", "thought": True, "thoughtSignature": "sig_thought_1"},
        {"text": "[[ ## answer ## ]]\nFirst ", "thoughtSignature": "sig_part_1"},
        {"text": "second\n\n[[ ## completed ## ]]", "thoughtSignature": "sig_part_2"},
    ]

    wire3 = json.loads(transport.requests[2].body)
    assert wire3["contents"][1]["parts"] == [
        {"text": "plan", "thought": True, "thoughtSignature": "sig_thought_1"},
        {"text": "[[ ## answer ## ]]\nManually edited\n\n[[ ## completed ## ]]\n", "thoughtSignature": "sig_part_1"},
    ]


def test_mixed_continuation_across_multiple_candidates_succeeds(monkeypatch):
    import json

    from tests.clients.test_lm_engine_execution import DualResponse, native_transport

    class ConvSig(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()

    cand_a = DualResponse(status=200, body=json.dumps({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": "[[ ## answer ## ]]\nCand A\n\n[[ ## completed ## ]]", "thoughtSignature": "sig_cand_0"}],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4, "totalTokenCount": 12},
    }).encode())
    cand_b = DualResponse(status=200, body=json.dumps({
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [{"text": "[[ ## answer ## ]]\nCand B\n\n[[ ## completed ## ]]"}],
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4, "totalTokenCount": 12},
    }).encode())
    followup_a = DualResponse(status=200, body=json.dumps({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "[[ ## answer ## ]]\nFollowup A\n\n[[ ## completed ## ]]"}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 14, "candidatesTokenCount": 4, "totalTokenCount": 18},
    }).encode())
    followup_b = DualResponse(status=200, body=json.dumps({
        "candidates": [{
            "content": {"role": "model", "parts": [{"text": "[[ ## answer ## ]]\nFollowup B\n\n[[ ## completed ## ]]"}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 14, "candidatesTokenCount": 4, "totalTokenCount": 18},
    }).encode())

    transport = native_transport(monkeypatch, [cand_a, cand_b, followup_a, followup_b])
    lm = dspy.LM("gemini/gemini-2.5-pro", engine="lm15", api_key="fake", cache=False, num_retries=0)
    predict = dspy.Predict(ConvSig)
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        pred = predict(question="Pick", history=dspy.History(messages=[]), config={"n": 2})
        assert len(pred.completions) == 2
        assert [c.answer for c in pred.completions] == ["Cand A", "Cand B"]

        hist_a = dspy.History(messages=[{"question": "Pick", **pred.completions[0]}])
        predict(question="Next A", history=hist_a)

        hist_b = dspy.History(messages=[{"question": "Pick", **pred.completions[1]}])
        predict(question="Next B", history=hist_b)

    wire_a = json.loads(transport.requests[2].body)
    assert wire_a["contents"][1]["parts"] == [
        {"text": "[[ ## answer ## ]]\nCand A\n\n[[ ## completed ## ]]\n", "thoughtSignature": "sig_cand_0"},
    ]
    wire_b = json.loads(transport.requests[3].body)
    assert wire_b["contents"][1]["parts"] == [
        {"text": "[[ ## answer ## ]]\nCand B\n\n[[ ## completed ## ]]\n"},
    ]

