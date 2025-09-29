import re

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
    dspy.settings.configure(lm=lm, adapter=adapter)

    react = dspy.ReAct("question -> answer", tools=[make_images])
    react(question="Draw me something red")

    sigs_with_obs = [sig for sig, inputs in captured_calls if "observation_0" in inputs]
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
    dspy.settings.configure(lm=lm)

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
    assert outputs.trajectory == expected_trajectory


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
    dspy.settings.configure(lm=lm)
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
    assert outputs.trajectory == expected_trajectory


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

    # Verify that older entries in the trajectory were truncated
    assert "thought_0" not in result.trajectory
    assert "thought_2" in result.trajectory
    assert result.output_text == "Final output"


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
    dspy.settings.configure(lm=lm)

    outputs = react(a=1, b=2, max_iters=2)
    traj = outputs.trajectory

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
    assert outputs.trajectory == expected_trajectory


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
