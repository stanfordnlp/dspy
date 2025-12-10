
import litellm
import pytest
from pydantic import BaseModel

import dspy
from dspy.utils.dummies import DummyLM


@pytest.mark.extra
def test_tool_observation_preserves_custom_type():
    pytest.importorskip("PIL.Image")
    from PIL import Image

    captured_calls = []

    class SpyChatAdapter(dspy.ChatAdapter):
        def format_user_message_content(self, signature, inputs, *args, **kwargs):
            captured_calls.append((signature, dict(inputs)))
            return super().format_user_message_content(signature, inputs, *args, **kwargs)

    def make_images():
        return dspy.Image("https://example.com/test.png"), dspy.Image(Image.new("RGB", (100, 100), "red"))


    adapter = SpyChatAdapter()
    lm = DummyLM(
        [
            {
                "next_thought": "I should call the image tool.",
                "next_tool_name": "make_images",
                "next_tool_args": {},
            },
            {
                "next_thought": "I now have the image so I can finish.",
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            {"reasoning": "image ready", "answer": "done"},
        ],
        adapter=adapter,
    )
    dspy.configure(lm=lm, adapter=adapter)

    react = dspy.ReAct("question -> answer", tools=[make_images])
    react(question="Draw me something red")

    sigs_with_obs = [sig for sig, inputs in captured_calls if "observation_0" in str(inputs)]
    assert sigs_with_obs, "Expected ReAct to format a trajectory containing observation_0"

    observation_content = lm.history[1]["messages"][1]["content"]
    assert sum(1 for part in observation_content if isinstance(part, dict) and part.get("type") == "image_url") == 2


def test_tool_calling_with_pydantic_args():
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: dict[str, str]

    def write_invitation_letter(participant_name: str, event_info: CalendarEvent):
        if participant_name not in event_info.participants:
            return None
        return f"It's my honor to invite {participant_name} to event {event_info.name} on {event_info.date}"

    class InvitationSignature(dspy.Signature):
        participant_name: str = dspy.InputField(desc="The name of the participant to invite")
        event_info: CalendarEvent = dspy.InputField(desc="The information about the event")
        invitation_letter: str = dspy.OutputField(desc="The invitation letter to be sent to the participant")

    react = dspy.ReAct(InvitationSignature, tools=[write_invitation_letter])

    lm = DummyLM(
        [
            {
                "next_thought": "I need to write an invitation letter for Alice to the Science Fair event.",
                "next_tool_name": "write_invitation_letter",
                "next_tool_args": {
                    "participant_name": "Alice",
                    "event_info": {
                        "name": "Science Fair",
                        "date": "Friday",
                        "participants": {"Alice": "female", "Bob": "male"},
                    },
                },
            },
            {
                "next_thought": (
                    "I have successfully written the invitation letter for Alice to the Science Fair. Now "
                    "I can finish the task."
                ),
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            {
                "reasoning": "This is a very rigorous reasoning process, trust me bro!",
                "invitation_letter": "It's my honor to invite Alice to the Science Fair event on Friday.",
            },
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(
        participant_name="Alice",
        event_info=CalendarEvent(
            name="Science Fair",
            date="Friday",
            participants={"Alice": "female", "Bob": "male"},
        ),
    )
    assert outputs.invitation_letter == "It's my honor to invite Alice to the Science Fair event on Friday."

    # Verify trajectory is a History object with raw mode (tool call format)
    traj = outputs.trajectory
    assert isinstance(traj, dspy.History)
    assert traj.mode == "raw"
    # 2 tool calls (write_invitation_letter + finish), each = assistant + tool = 4 messages + 1 extract = 5 messages
    assert len(traj.messages) == 5

    # Check first message (tool call - assistant)
    msg0 = traj.messages[0]
    assert msg0["role"] == "assistant"
    assert msg0["content"] == "I need to write an invitation letter for Alice to the Science Fair event."
    assert len(msg0["tool_calls"]) == 1
    assert msg0["tool_calls"][0]["function"]["name"] == "write_invitation_letter"

    # Check second message (tool response)
    msg1 = traj.messages[1]
    assert msg1["role"] == "tool"
    assert msg1["tool_call_id"] == msg0["tool_calls"][0]["id"]
    assert "It's my honor to invite Alice to event Science Fair on Friday" in msg1["content"]

    # Check third message (finish - assistant)
    msg2 = traj.messages[2]
    assert msg2["role"] == "assistant"
    assert msg2["tool_calls"][0]["function"]["name"] == "finish"

    # Check fourth message (finish - tool response)
    msg3 = traj.messages[3]
    assert msg3["role"] == "tool"
    assert msg3["content"] == "Completed."

    # Check last message (extract)
    msg_extract = traj.messages[-1]
    assert msg_extract["role"] == "assistant"
    assert "This is a very rigorous reasoning process, trust me bro!" in msg_extract["content"]
    assert "invitation_letter" in msg_extract["content"]


def test_tool_calling_without_typehint():
    def foo(a, b):
        """Add two numbers."""
        return a + b

    react = dspy.ReAct("a, b -> c:int", tools=[foo])
    lm = DummyLM(
        [
            {"next_thought": "I need to add two numbers.", "next_tool_name": "foo", "next_tool_args": {"a": 1, "b": 2}},
            {"next_thought": "I have the sum, now I can finish.", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "I added the numbers successfully", "c": 3},
        ]
    )
    dspy.configure(lm=lm)
    outputs = react(a=1, b=2)

    # Verify trajectory is a History object with raw mode
    traj = outputs.trajectory
    assert isinstance(traj, dspy.History)
    assert traj.mode == "raw"
    # 2 tool calls (each = assistant + tool) + 1 extract = 5 messages
    assert len(traj.messages) == 5

    # Check first message (tool call - assistant)
    msg0 = traj.messages[0]
    assert msg0["role"] == "assistant"
    assert msg0["content"] == "I need to add two numbers."
    assert msg0["tool_calls"][0]["function"]["name"] == "foo"

    # Check second message (tool response)
    msg1 = traj.messages[1]
    assert msg1["role"] == "tool"
    assert msg1["content"] == "3"  # JSON serialized

    # Check third message (finish - assistant)
    msg2 = traj.messages[2]
    assert msg2["role"] == "assistant"
    assert msg2["content"] == "I have the sum, now I can finish."
    assert msg2["tool_calls"][0]["function"]["name"] == "finish"

    # Check fourth message (finish - tool response)
    msg3 = traj.messages[3]
    assert msg3["role"] == "tool"
    assert msg3["content"] == "Completed."

    # Check last message (extract)
    msg_extract = traj.messages[-1]
    assert msg_extract["role"] == "assistant"
    assert "I added the numbers successfully" in msg_extract["content"]
    assert "c: 3" in msg_extract["content"]


def test_trajectory_truncation():
    # Create a simple tool for testing
    def echo(text: str) -> str:
        return f"Echoed: {text}"

    # Create ReAct instance with our echo tool
    react = dspy.ReAct("input_text -> output_text", tools=[echo])

    # Mock react.react to simulate multiple tool calls
    call_count = 0

    def mock_react(**kwargs):
        nonlocal call_count
        call_count += 1

        if call_count < 3:
            # First 2 calls use the echo tool
            return dspy.Prediction(
                next_thought=f"Thought {call_count}",
                next_tool_name="echo",
                next_tool_args={"text": f"Text {call_count}"},
            )
        elif call_count == 3:
            # The 3rd call raises context window exceeded error
            raise litellm.ContextWindowExceededError("Context window exceeded", "dummy_model", "dummy_provider")
        else:
            # The 4th call finishes
            return dspy.Prediction(next_thought="Final thought", next_tool_name="finish", next_tool_args={})

    react.react = mock_react
    react.extract = lambda **kwargs: dspy.Prediction(output_text="Final output")

    # Call forward and get the result
    result = react(input_text="test input")

    # Verify trajectory is a History object
    traj = result.trajectory
    assert isinstance(traj, dspy.History)
    assert traj.mode == "raw"

    # Verify that older entries were truncated (first assistant+tool pair removed)
    # After truncation, we should have messages for: Thought 2 (assistant+tool), finish (assistant+tool), extract
    assert len(traj.messages) >= 4

    # First message should be Thought 2's assistant message (Thought 1 was truncated)
    assert traj.messages[0]["role"] == "assistant"
    assert traj.messages[0]["content"] == "Thought 2"

    assert result.output_text == "Final output"


@pytest.mark.asyncio
async def test_context_window_exceeded_after_retries():
    def echo(text: str) -> str:
        return f"Echoed: {text}"

    react = dspy.ReAct("input_text -> output_text", tools=[echo])

    def mock_react(**kwargs):
        raise litellm.ContextWindowExceededError("Context window exceeded", "dummy_model", "dummy_provider")

    # Test sync version
    extract_calls = []

    def mock_extract(**kwargs):
        extract_calls.append(kwargs)
        return dspy.Prediction(output_text="Fallback output")

    react.react = mock_react
    react.extract = mock_extract

    result = react(input_text="test input")
    assert result.trajectory == {}
    assert result.output_text == "Fallback output"
    assert len(extract_calls) == 1
    assert extract_calls[0]["input_text"] == "test input"
    assert "trajectory" in extract_calls[0]

    # Test async version
    async_extract_calls = []

    async def mock_react_async(**kwargs):
        raise litellm.ContextWindowExceededError("Context window exceeded", "dummy_model", "dummy_provider")

    async def mock_extract_async(**kwargs):
        async_extract_calls.append(kwargs)
        return dspy.Prediction(output_text="Fallback output")

    react.react.acall = mock_react_async
    react.extract.acall = mock_extract_async

    result = await react.acall(input_text="test input")
    assert result.trajectory == {}
    assert result.output_text == "Fallback output"
    assert len(async_extract_calls) == 1
    assert async_extract_calls[0]["input_text"] == "test input"
    assert "trajectory" in async_extract_calls[0]


def test_error_retry():
    # --- a tiny tool that always fails -------------------------------------
    def foo(a, b):
        raise Exception("tool error")

    # --- program under test -------------------------------------------------
    react = dspy.ReAct("a, b -> c:int", tools=[foo])
    lm = DummyLM(
        [
            {
                "next_thought": "I need to add two numbers.",
                "next_tool_name": "foo",
                "next_tool_args": {"a": 1, "b": 2},
            },
            {
                "next_thought": "I need to add two numbers.",
                "next_tool_name": "foo",
                "next_tool_args": {"a": 1, "b": 2},
            },
            # (The model *would* succeed on the 3rd turn, but max_iters=2 stops earlier.)
            {"reasoning": "I added the numbers successfully", "c": 3},
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(a=1, b=2, max_iters=2)
    traj = outputs.trajectory

    # Verify trajectory is a History object with raw mode
    assert isinstance(traj, dspy.History)
    assert traj.mode == "raw"
    # 2 tool calls (each = assistant + tool) + 1 extract = 5 messages
    assert len(traj.messages) == 5

    # Check tool call messages have the expected structure
    # Messages 0, 2 are assistant messages with tool_calls
    # Messages 1, 3 are tool response messages
    for i in [0, 2]:
        msg = traj.messages[i]
        assert msg["role"] == "assistant"
        assert msg["content"] == "I need to add two numbers."
        assert msg["tool_calls"][0]["function"]["name"] == "foo"

    for i in [1, 3]:
        msg = traj.messages[i]
        assert msg["role"] == "tool"
        # Observation should contain the error
        assert "tool error" in msg["content"]

    # Check extract message
    msg_extract = traj.messages[-1]
    assert msg_extract["role"] == "assistant"
    assert "I added the numbers successfully" in msg_extract["content"]
    assert "c: 3" in msg_extract["content"]


@pytest.mark.asyncio
async def test_async_tool_calling_with_pydantic_args():
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: dict[str, str]

    async def write_invitation_letter(participant_name: str, event_info: CalendarEvent):
        if participant_name not in event_info.participants:
            return None
        return f"It's my honor to invite {participant_name} to event {event_info.name} on {event_info.date}"

    class InvitationSignature(dspy.Signature):
        participant_name: str = dspy.InputField(desc="The name of the participant to invite")
        event_info: CalendarEvent = dspy.InputField(desc="The information about the event")
        invitation_letter: str = dspy.OutputField(desc="The invitation letter to be sent to the participant")

    react = dspy.ReAct(InvitationSignature, tools=[write_invitation_letter])

    lm = DummyLM(
        [
            {
                "next_thought": "I need to write an invitation letter for Alice to the Science Fair event.",
                "next_tool_name": "write_invitation_letter",
                "next_tool_args": {
                    "participant_name": "Alice",
                    "event_info": {
                        "name": "Science Fair",
                        "date": "Friday",
                        "participants": {"Alice": "female", "Bob": "male"},
                    },
                },
            },
            {
                "next_thought": (
                    "I have successfully written the invitation letter for Alice to the Science Fair. Now "
                    "I can finish the task."
                ),
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            {
                "reasoning": "This is a very rigorous reasoning process, trust me bro!",
                "invitation_letter": "It's my honor to invite Alice to the Science Fair event on Friday.",
            },
        ]
    )
    with dspy.context(lm=lm):
        outputs = await react.acall(
            participant_name="Alice",
            event_info=CalendarEvent(
                name="Science Fair",
                date="Friday",
                participants={"Alice": "female", "Bob": "male"},
            ),
        )
    assert outputs.invitation_letter == "It's my honor to invite Alice to the Science Fair event on Friday."

    # Verify trajectory is a History object with raw mode
    traj = outputs.trajectory
    assert isinstance(traj, dspy.History)
    assert traj.mode == "raw"
    # 2 tool calls (write_invitation_letter + finish), each = assistant + tool = 4 messages + 1 extract = 5 messages
    assert len(traj.messages) == 5

    # Check first message (tool call - assistant)
    msg0 = traj.messages[0]
    assert msg0["role"] == "assistant"
    assert msg0["tool_calls"][0]["function"]["name"] == "write_invitation_letter"

    # Check second message (tool response)
    msg1 = traj.messages[1]
    assert msg1["role"] == "tool"
    assert "It's my honor to invite Alice to event Science Fair on Friday" in msg1["content"]

    # Check third message (finish - assistant)
    msg2 = traj.messages[2]
    assert msg2["role"] == "assistant"
    assert msg2["tool_calls"][0]["function"]["name"] == "finish"

    # Check fourth message (finish - tool response)
    msg3 = traj.messages[3]
    assert msg3["role"] == "tool"
    assert msg3["content"] == "Completed."

    # Check last message (extract)
    msg_extract = traj.messages[-1]
    assert msg_extract["role"] == "assistant"
    assert "This is a very rigorous reasoning process, trust me bro!" in msg_extract["content"]


@pytest.mark.asyncio
async def test_async_error_retry():
    # A tiny tool that always fails
    async def foo(a, b):
        raise Exception("tool error")

    react = dspy.ReAct("a, b -> c:int", tools=[foo])
    lm = DummyLM(
        [
            {
                "next_thought": "I need to add two numbers.",
                "next_tool_name": "foo",
                "next_tool_args": {"a": 1, "b": 2},
            },
            {
                "next_thought": "I need to add two numbers.",
                "next_tool_name": "foo",
                "next_tool_args": {"a": 1, "b": 2},
            },
            # (The model *would* succeed on the 3rd turn, but max_iters=2 stops earlier.)
            {"reasoning": "I added the numbers successfully", "c": 3},
        ]
    )
    with dspy.context(lm=lm):
        outputs = await react.acall(a=1, b=2, max_iters=2)
    traj = outputs.trajectory

    # Verify trajectory is a History object with raw mode
    assert isinstance(traj, dspy.History)
    assert traj.mode == "raw"
    # 2 tool calls (each = assistant + tool) + 1 extract = 5 messages
    assert len(traj.messages) == 5

    # Check tool call messages have the expected structure
    for i in [0, 2]:
        msg = traj.messages[i]
        assert msg["role"] == "assistant"
        assert msg["content"] == "I need to add two numbers."
        assert msg["tool_calls"][0]["function"]["name"] == "foo"

    for i in [1, 3]:
        msg = traj.messages[i]
        assert msg["role"] == "tool"
        # Observation should contain the error
        assert "tool error" in msg["content"]

    # Check extract message
    msg_extract = traj.messages[-1]
    assert msg_extract["role"] == "assistant"
    assert "I added the numbers successfully" in msg_extract["content"]
    assert "c: 3" in msg_extract["content"]
