import re

import litellm
import pytest
from pydantic import BaseModel

import dspy
from dspy.utils.dummies import DummyLM


def _history_to_trajectory(history: dspy.History) -> dict[str, object]:
    trajectory = {}
    turn_idx = 0
    messages = history.messages
    i = 0
    while i < len(messages):
        message = messages[i]
        if message.role != "assistant":
            i += 1
            continue

        fields = message.fields
        if not {"next_thought", "next_tool_name", "next_tool_args"}.issubset(fields):
            i += 1
            continue

        trajectory[f"thought_{turn_idx}"] = fields["next_thought"]
        trajectory[f"tool_name_{turn_idx}"] = fields["next_tool_name"]
        trajectory[f"tool_args_{turn_idx}"] = fields["next_tool_args"]

        if i + 1 < len(messages) and messages[i + 1].role == "tool":
            trajectory[f"observation_{turn_idx}"] = messages[i + 1].fields.get("observation")
            i += 2
        else:
            i += 1

        turn_idx += 1

    return trajectory


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

    sigs_with_obs = [sig for sig, inputs in captured_calls if "observation" in inputs]
    assert sigs_with_obs, "Expected ReAct to include `observation` in later turns"
    assert all("question" not in inputs for _, inputs in captured_calls if "observation" in inputs)
    assert all("question" not in inputs for _, inputs in captured_calls if "next_thought" in inputs)

    second_turn_messages = lm.history[1]["messages"]
    observation_content = next(msg["content"] for msg in second_turn_messages if isinstance(msg["content"], list))
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

    expected_trajectory = {
        "thought_0": "I need to write an invitation letter for Alice to the Science Fair event.",
        "tool_name_0": "write_invitation_letter",
        "tool_args_0": {
            "participant_name": "Alice",
            "event_info": {
                "name": "Science Fair",
                "date": "Friday",
                "participants": {"Alice": "female", "Bob": "male"},
            },
        },
        "observation_0": "It's my honor to invite Alice to event Science Fair on Friday",
        "thought_1": "I have successfully written the invitation letter for Alice to the Science Fair. Now I can finish the task.",
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": "Completed.",
    }
    assert _history_to_trajectory(outputs.history) == expected_trajectory


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

    expected_trajectory = {
        "thought_0": "I need to add two numbers.",
        "tool_name_0": "foo",
        "tool_args_0": {
            "a": 1,
            "b": 2,
        },
        "observation_0": 3,
        "thought_1": "I have the sum, now I can finish.",
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": "Completed.",
    }
    assert _history_to_trajectory(outputs.history) == expected_trajectory


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

    with pytest.raises(litellm.ContextWindowExceededError):
        react(input_text="test input")


@pytest.mark.asyncio
async def test_context_window_exceeded_propagates():
    def echo(text: str) -> str:
        return f"Echoed: {text}"

    react = dspy.ReAct("input_text -> output_text", tools=[echo])

    def mock_react(**kwargs):
        raise litellm.ContextWindowExceededError("Context window exceeded", "dummy_model", "dummy_provider")

    react.react = mock_react
    react.extract = lambda **kwargs: dspy.Prediction(output_text="Fallback output")

    with pytest.raises(litellm.ContextWindowExceededError):
        react(input_text="test input")

    # Test async version
    async def mock_react_async(**kwargs):
        raise litellm.ContextWindowExceededError("Context window exceeded", "dummy_model", "dummy_provider")

    react.react.acall = mock_react_async
    react.extract.acall = lambda **kwargs: dspy.Prediction(output_text="Fallback output")

    with pytest.raises(litellm.ContextWindowExceededError):
        await react.acall(input_text="test input")


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
    traj = _history_to_trajectory(outputs.history)

    # --- exact-match checks (thoughts + tool calls) -------------------------
    control_expected = {
        "thought_0": "I need to add two numbers.",
        "tool_name_0": "foo",
        "tool_args_0": {"a": 1, "b": 2},
        "thought_1": "I need to add two numbers.",
        "tool_name_1": "foo",
        "tool_args_1": {"a": 1, "b": 2},
    }
    for k, v in control_expected.items():
        assert traj[k] == v, f"{k} mismatch"

    # --- flexible checks for observations ----------------------------------
    # We only care that each observation mentions our error string; we ignore
    # any extra traceback detail or differing prefixes.
    for i in range(2):
        obs = traj[f"observation_{i}"]
        assert re.search(r"\btool error\b", obs), f"unexpected observation_{i!r}: {obs}"


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

    expected_trajectory = {
        "thought_0": "I need to write an invitation letter for Alice to the Science Fair event.",
        "tool_name_0": "write_invitation_letter",
        "tool_args_0": {
            "participant_name": "Alice",
            "event_info": {
                "name": "Science Fair",
                "date": "Friday",
                "participants": {"Alice": "female", "Bob": "male"},
            },
        },
        "observation_0": "It's my honor to invite Alice to event Science Fair on Friday",
        "thought_1": "I have successfully written the invitation letter for Alice to the Science Fair. Now I can finish the task.",
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": "Completed.",
    }
    assert _history_to_trajectory(outputs.history) == expected_trajectory


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
    traj = _history_to_trajectory(outputs.history)

    # Exact-match checks (thoughts + tool calls)
    control_expected = {
        "thought_0": "I need to add two numbers.",
        "tool_name_0": "foo",
        "tool_args_0": {"a": 1, "b": 2},
        "thought_1": "I need to add two numbers.",
        "tool_name_1": "foo",
        "tool_args_1": {"a": 1, "b": 2},
    }
    for k, v in control_expected.items():
        assert traj[k] == v, f"{k} mismatch"

    # Flexible checks for observations
    # We only care that each observation mentions our error string; we ignore
    # any extra traceback detail or differing prefixes.
    for i in range(2):
        obs = traj[f"observation_{i}"]
        assert re.search(r"\btool error\b", obs), f"unexpected observation_{i!r}: {obs}"


def test_resume_strict_from_completed_history():
    def foo(a, b):
        return a + b

    react = dspy.ReAct("a, b -> c:int", tools=[foo])
    history = dspy.History(
        messages=[
            dspy.HistoryMessage(role="user", fields={"a": 1, "b": 2}),
            dspy.HistoryMessage(
                role="assistant",
                fields={
                    "next_thought": "Add the numbers first.",
                    "next_tool_name": "foo",
                    "next_tool_args": {"a": 1, "b": 2},
                },
            ),
            dspy.HistoryMessage(role="tool", fields={"tool_name": "foo", "observation": 3}),
        ]
    )

    lm = DummyLM(
        [
            {"next_thought": "Now I can finish.", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "We already computed the sum", "c": 3},
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(a=1, b=2, history=history, resume="strict", max_iters=3)
    assert outputs.c == 3
    trajectory = _history_to_trajectory(outputs.history)
    assert trajectory["thought_0"] == "Add the numbers first."
    assert trajectory["observation_0"] == 3
    assert trajectory["tool_name_1"] == "finish"
    assert trajectory["observation_1"] == "Completed."


def test_resume_executes_pending_tool_call():
    def foo(a, b):
        return a + b

    react = dspy.ReAct("a, b -> c:int", tools=[foo])
    history = dspy.History(
        messages=[
            dspy.HistoryMessage(role="user", fields={"a": 1, "b": 2}),
            dspy.HistoryMessage(
                role="assistant",
                fields={
                    "next_thought": "I should add the two inputs.",
                    "next_tool_name": "foo",
                    "next_tool_args": {"a": 1, "b": 2},
                },
            ),
        ]
    )

    lm = DummyLM(
        [
            {"next_thought": "Now I can finish.", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "The pending call was completed before continuing", "c": 3},
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(a=1, b=2, history=history, resume="strict", max_iters=3)
    assert outputs.c == 3
    trajectory = _history_to_trajectory(outputs.history)
    assert trajectory["tool_name_0"] == "foo"
    assert trajectory["observation_0"] == 3


def test_resume_strict_rejects_input_mismatch():
    def foo(a, b):
        return a + b

    react = dspy.ReAct("a, b -> c:int", tools=[foo])
    history = dspy.History(messages=[dspy.HistoryMessage(role="user", fields={"a": 9, "b": 2})])
    dspy.configure(lm=DummyLM([]))

    with pytest.raises(ValueError, match="does not match current input"):
        react(a=1, b=2, history=history, resume="strict")
