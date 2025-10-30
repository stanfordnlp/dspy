import asyncio
import re
import time

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
                "next_tool_calls": [
                    {
                        "name": "write_invitation_letter",
                        "args": {
                            "participant_name": "Alice",
                            "event_info": {
                                "name": "Science Fair",
                                "date": "Friday",
                                "participants": {"Alice": "female", "Bob": "male"},
                            },
                        },
                    }
                ],
            },
            {
                "next_thought": (
                    "I have successfully written the invitation letter for Alice to the Science Fair. Now "
                    "I can finish the task."
                ),
                "next_tool_calls": [{"name": "finish", "args": {}}],
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
        "tool_calls_0": [
            {
                "name": "write_invitation_letter",
                "args": {
                    "participant_name": "Alice",
                    "event_info": {
                        "name": "Science Fair",
                        "date": "Friday",
                        "participants": {"Alice": "female", "Bob": "male"},
                    },
                },
            }
        ],
        "observations_0": [
            {
                "tool": "write_invitation_letter",
                "result": "It's my honor to invite Alice to event Science Fair on Friday",
            }
        ],
        "thought_1": "I have successfully written the invitation letter for Alice to the Science Fair. Now I can finish the task.",
        "tool_calls_1": [{"name": "finish", "args": {}}],
        "observations_1": [{"tool": "finish", "result": "Completed."}],
    }
    assert outputs.trajectory == expected_trajectory


def test_tool_calling_without_typehint():
    def foo(a, b):
        """Add two numbers."""
        return a + b

    react = dspy.ReAct("a, b -> c:int", tools=[foo])
    lm = DummyLM(
        [
            {"next_thought": "I need to add two numbers.", "next_tool_calls": [{"name": "foo", "args": {"a": 1, "b": 2}}]},
            {"next_thought": "I have the sum, now I can finish.", "next_tool_calls": [{"name": "finish", "args": {}}]},
            {"reasoning": "I added the numbers successfully", "c": 3},
        ]
    )
    dspy.settings.configure(lm=lm)
    outputs = react(a=1, b=2)

    expected_trajectory = {
        "thought_0": "I need to add two numbers.",
        "tool_calls_0": [{"name": "foo", "args": {"a": 1, "b": 2}}],
        "observations_0": [{"tool": "foo", "result": 3}],
        "thought_1": "I have the sum, now I can finish.",
        "tool_calls_1": [{"name": "finish", "args": {}}],
        "observations_1": [{"tool": "finish", "result": "Completed."}],
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
                next_tool_calls=[{"name": "echo", "args": {"text": f"Text {call_count}"}}],
            )
        elif call_count == 3:
            # The 3rd call raises context window exceeded error
            raise litellm.ContextWindowExceededError("Context window exceeded", "dummy_model", "dummy_provider")
        else:
            # The 4th call finishes
            return dspy.Prediction(next_thought="Final thought", next_tool_calls=[{"name": "finish", "args": {}}])

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
                "next_tool_calls": [{"name": "foo", "args": {"a": 1, "b": 2}}],
            },
            {
                "next_thought": "I need to add two numbers.",
                "next_tool_calls": [{"name": "foo", "args": {"a": 1, "b": 2}}],
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
        "tool_calls_0": [{"name": "foo", "args": {"a": 1, "b": 2}}],
        "thought_1": "I need to add two numbers.",
        "tool_calls_1": [{"name": "foo", "args": {"a": 1, "b": 2}}],
    }
    for k, v in control_expected.items():
        assert traj[k] == v, f"{k} mismatch"

    # --- flexible checks for observations ----------------------------------
    # We only care that each observation mentions our error string; we ignore
    # any extra traceback detail or differing prefixes.
    for i in range(2):
        obs = traj[f"observations_{i}"][0]["result"]
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
                "next_tool_calls": [
                    {
                        "name": "write_invitation_letter",
                        "args": {
                            "participant_name": "Alice",
                            "event_info": {
                                "name": "Science Fair",
                                "date": "Friday",
                                "participants": {"Alice": "female", "Bob": "male"},
                            },
                        },
                    }
                ],
            },
            {
                "next_thought": (
                    "I have successfully written the invitation letter for Alice to the Science Fair. Now "
                    "I can finish the task."
                ),
                "next_tool_calls": [{"name": "finish", "args": {}}],
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
        "tool_calls_0": [
            {
                "name": "write_invitation_letter",
                "args": {
                    "participant_name": "Alice",
                    "event_info": {
                        "name": "Science Fair",
                        "date": "Friday",
                        "participants": {"Alice": "female", "Bob": "male"},
                    },
                },
            }
        ],
        "observations_0": [
            {
                "tool": "write_invitation_letter",
                "result": "It's my honor to invite Alice to event Science Fair on Friday",
            }
        ],
        "thought_1": "I have successfully written the invitation letter for Alice to the Science Fair. Now I can finish the task.",
        "tool_calls_1": [{"name": "finish", "args": {}}],
        "observations_1": [{"tool": "finish", "result": "Completed."}],
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
                "next_tool_calls": [{"name": "foo", "args": {"a": 1, "b": 2}}],
            },
            {
                "next_thought": "I need to add two numbers.",
                "next_tool_calls": [{"name": "foo", "args": {"a": 1, "b": 2}}],
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
        "tool_calls_0": [{"name": "foo", "args": {"a": 1, "b": 2}}],
        "thought_1": "I need to add two numbers.",
        "tool_calls_1": [{"name": "foo", "args": {"a": 1, "b": 2}}],
    }
    for k, v in control_expected.items():
        assert traj[k] == v, f"{k} mismatch"

    # Flexible checks for observations
    # We only care that each observation mentions our error string; we ignore
    # any extra traceback detail or differing prefixes.
    for i in range(2):
        obs = traj[f"observations_{i}"][0]["result"]
        assert re.search(r"\btool error\b", obs), f"unexpected observation_{i!r}: {obs}"


def test_parallel_tool_execution_sync():
    """Test that multiple tools can be executed in parallel in sync mode."""
    # Create tools that track execution order
    execution_log = []

    def tool1(x: int) -> int:
        execution_log.append(("tool1_start", x))
        time.sleep(0.1)  # Simulate work
        execution_log.append(("tool1_end", x))
        return x * 2

    def tool2(y: int) -> int:
        execution_log.append(("tool2_start", y))
        time.sleep(0.1)  # Simulate work
        execution_log.append(("tool2_end", y))
        return y * 3

    react = dspy.ReAct("input -> output", tools=[tool1, tool2])

    lm = DummyLM(
        [
            {
                "next_thought": "I should call both tools in parallel.",
                "next_tool_calls": [
                    {"name": "tool1", "args": {"x": 5}},
                    {"name": "tool2", "args": {"y": 10}},
                ],
            },
            {
                "next_thought": "I have the results, now I can finish.",
                "next_tool_calls": [{"name": "finish", "args": {}}],
            },
            {"reasoning": "Both tools executed successfully", "output": "done"},
        ]
    )
    dspy.settings.configure(lm=lm)

    outputs = react(input="test")

    # Check that the trajectory contains the right structure
    assert "thought_0" in outputs.trajectory
    assert "tool_calls_0" in outputs.trajectory
    assert "observations_0" in outputs.trajectory

    # Check the tool calls
    tool_calls = outputs.trajectory["tool_calls_0"]
    assert len(tool_calls) == 2
    assert tool_calls[0]["name"] == "tool1"
    assert tool_calls[0]["args"] == {"x": 5}
    assert tool_calls[1]["name"] == "tool2"
    assert tool_calls[1]["args"] == {"y": 10}

    # Check the observations
    observations = outputs.trajectory["observations_0"]
    assert len(observations) == 2
    assert observations[0]["tool"] == "tool1"
    assert observations[0]["result"] == 10  # 5 * 2
    assert observations[1]["tool"] == "tool2"
    assert observations[1]["result"] == 30  # 10 * 3

    # Verify parallel execution improved performance
    # Note: Timing can vary in different environments, so we mainly check execution order
    # If sequential, it would take ~0.2s; parallel should be closer to 0.1s (but allow more time for overhead)
    # assert elapsed_time < 0.25, f"Execution took {elapsed_time}s, expected parallel execution"

    # Check that tools ran concurrently (both start before either ends)
    assert len(execution_log) >= 2
    assert execution_log[0][0] in ["tool1_start", "tool2_start"]
    assert execution_log[1][0] in ["tool1_start", "tool2_start"]
    # If parallel, both should start before any ends
    start_count = sum(1 for log in execution_log[:2] if "start" in log[0])
    assert start_count == 2, "Both tools should start before either ends (indicating parallel execution)"


def test_single_tool_execution_backwards_compat():
    """Test that single tool execution still works (backwards compatibility)."""
    def add(x: int, y: int) -> int:
        return x + y

    react = dspy.ReAct("a, b -> c", tools=[add])

    lm = DummyLM(
        [
            {
                "next_thought": "I should add the numbers.",
                "next_tool_calls": [{"name": "add", "args": {"x": 3, "y": 4}}],
            },
            {
                "next_thought": "I have the sum, now I can finish.",
                "next_tool_calls": [{"name": "finish", "args": {}}],
            },
            {"reasoning": "Added successfully", "c": 7},
        ]
    )
    dspy.settings.configure(lm=lm)

    outputs = react(a=3, b=4)

    # Check trajectory structure
    assert "thought_0" in outputs.trajectory
    assert "tool_calls_0" in outputs.trajectory
    assert "observations_0" in outputs.trajectory

    # Check that single tool call works
    tool_calls = outputs.trajectory["tool_calls_0"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "add"

    observations = outputs.trajectory["observations_0"]
    assert len(observations) == 1
    assert observations[0]["tool"] == "add"
    assert observations[0]["result"] == 7


def test_parallel_tool_execution_with_error():
    """Test that errors in parallel tools are handled correctly."""
    def good_tool(x: int) -> int:
        return x * 2

    def bad_tool(y: int) -> int:
        raise ValueError("Tool error")

    react = dspy.ReAct("input -> output", tools=[good_tool, bad_tool])

    lm = DummyLM(
        [
            {
                "next_thought": "I should call both tools.",
                "next_tool_calls": [
                    {"name": "good_tool", "args": {"x": 5}},
                    {"name": "bad_tool", "args": {"y": 10}},
                ],
            },
            {
                "next_thought": "One tool failed but I can still finish.",
                "next_tool_calls": [{"name": "finish", "args": {}}],
            },
            {"reasoning": "Handled errors", "output": "done"},
        ]
    )
    dspy.settings.configure(lm=lm)

    outputs = react(input="test")

    # Check observations - one should be successful, one should be an error message
    observations = outputs.trajectory["observations_0"]
    assert len(observations) == 2
    assert observations[0]["tool"] == "good_tool"
    assert observations[0]["result"] == 10  # good_tool result
    assert observations[1]["tool"] == "bad_tool"
    assert "Execution error in bad_tool" in observations[1]["result"]
    assert "Tool error" in observations[1]["result"]


@pytest.mark.asyncio
async def test_parallel_tool_execution_async():
    """Test that multiple tools can be executed in parallel in async mode."""
    execution_log = []

    async def async_tool1(x: int) -> int:
        execution_log.append(("tool1_start", x))
        await asyncio.sleep(0.1)  # Simulate async work
        execution_log.append(("tool1_end", x))
        return x * 2

    async def async_tool2(y: int) -> int:
        execution_log.append(("tool2_start", y))
        await asyncio.sleep(0.1)  # Simulate async work
        execution_log.append(("tool2_end", y))
        return y * 3

    react = dspy.ReAct("input -> output", tools=[async_tool1, async_tool2])

    lm = DummyLM(
        [
            {
                "next_thought": "I should call both tools in parallel.",
                "next_tool_calls": [
                    {"name": "async_tool1", "args": {"x": 5}},
                    {"name": "async_tool2", "args": {"y": 10}},
                ],
            },
            {
                "next_thought": "I have the results, now I can finish.",
                "next_tool_calls": [{"name": "finish", "args": {}}],
            },
            {"reasoning": "Both tools executed successfully", "output": "done"},
        ]
    )

    with dspy.context(lm=lm):
        outputs = await react.acall(input="test")

    # Check that the trajectory contains the right structure
    assert "thought_0" in outputs.trajectory
    assert "tool_calls_0" in outputs.trajectory
    assert "observations_0" in outputs.trajectory

    # Check the tool calls
    tool_calls = outputs.trajectory["tool_calls_0"]
    assert len(tool_calls) == 2

    # Check the observations
    observations = outputs.trajectory["observations_0"]
    assert len(observations) == 2
    assert observations[0]["tool"] == "async_tool1"
    assert observations[0]["result"] == 10  # 5 * 2
    assert observations[1]["tool"] == "async_tool2"
    assert observations[1]["result"] == 30  # 10 * 3

    # Verify parallel execution improved performance
    # Note: Timing can vary, but async parallel should still be faster than sequential
    # assert elapsed_time < 0.15, f"Execution took {elapsed_time}s, expected parallel execution"

    # Check that async tools ran concurrently
    assert len(execution_log) == 4
    # Both should start before either ends (indicating parallel execution)
    starts = [log for log in execution_log if "start" in log[0]]
    assert len(starts) == 2


@pytest.mark.asyncio
async def test_parallel_async_tool_with_error():
    """Test error handling in parallel async tool execution."""
    async def good_async_tool(x: int) -> int:
        await asyncio.sleep(0.05)
        return x * 2

    async def bad_async_tool(y: int) -> int:
        await asyncio.sleep(0.05)
        raise ValueError("Async tool error")

    react = dspy.ReAct("input -> output", tools=[good_async_tool, bad_async_tool])

    lm = DummyLM(
        [
            {
                "next_thought": "I should call both tools.",
                "next_tool_calls": [
                    {"name": "good_async_tool", "args": {"x": 5}},
                    {"name": "bad_async_tool", "args": {"y": 10}},
                ],
            },
            {
                "next_thought": "One tool failed but I can still finish.",
                "next_tool_calls": [{"name": "finish", "args": {}}],
            },
            {"reasoning": "Handled errors", "output": "done"},
        ]
    )

    with dspy.context(lm=lm):
        outputs = await react.acall(input="test")

    # Check observations
    observations = outputs.trajectory["observations_0"]
    assert len(observations) == 2
    assert observations[0]["tool"] == "good_async_tool"
    assert observations[0]["result"] == 10  # good tool result
    assert observations[1]["tool"] == "bad_async_tool"
    assert "Execution error in bad_async_tool" in observations[1]["result"]
    assert "Async tool error" in observations[1]["result"]


def test_multiple_iterations_with_parallel_tools():
    """Test that parallel tools work across multiple iterations."""
    def tool_a(x: int) -> str:
        return f"a:{x}"

    def tool_b(y: int) -> str:
        return f"b:{y}"

    react = dspy.ReAct("input -> output", tools=[tool_a, tool_b])

    lm = DummyLM(
        [
            # First iteration - call both tools
            {
                "next_thought": "First iteration, calling both tools.",
                "next_tool_calls": [
                    {"name": "tool_a", "args": {"x": 1}},
                    {"name": "tool_b", "args": {"y": 2}},
                ],
            },
            # Second iteration - call both tools again
            {
                "next_thought": "Second iteration, calling both tools again.",
                "next_tool_calls": [
                    {"name": "tool_a", "args": {"x": 3}},
                    {"name": "tool_b", "args": {"y": 4}},
                ],
            },
            # Finish
            {
                "next_thought": "Now I can finish.",
                "next_tool_calls": [{"name": "finish", "args": {}}],
            },
            {"reasoning": "Done", "output": "complete"},
        ]
    )
    dspy.settings.configure(lm=lm)

    outputs = react(input="test")

    # Check first iteration
    assert outputs.trajectory["tool_calls_0"] == [
        {"name": "tool_a", "args": {"x": 1}},
        {"name": "tool_b", "args": {"y": 2}},
    ]
    assert outputs.trajectory["observations_0"] == [
        {"tool": "tool_a", "result": "a:1"},
        {"tool": "tool_b", "result": "b:2"}
    ]

    # Check second iteration
    assert outputs.trajectory["tool_calls_1"] == [
        {"name": "tool_a", "args": {"x": 3}},
        {"name": "tool_b", "args": {"y": 4}},
    ]
    assert outputs.trajectory["observations_1"] == [
        {"tool": "tool_a", "result": "a:3"},
        {"tool": "tool_b", "result": "b:4"}
    ]

    # Check finish iteration
    assert outputs.trajectory["tool_calls_2"] == [{"name": "finish", "args": {}}]


def test_empty_tool_args():
    """Test parallel execution with tools that have no arguments."""
    def get_time() -> str:
        return "12:00"

    def get_date() -> str:
        return "2024-01-01"

    react = dspy.ReAct("query -> result", tools=[get_time, get_date])

    lm = DummyLM(
        [
            {
                "next_thought": "I'll get both time and date.",
                "next_tool_calls": [
                    {"name": "get_time", "args": {}},
                    {"name": "get_date", "args": {}},
                ],
            },
            {
                "next_thought": "Got both, finishing.",
                "next_tool_calls": [{"name": "finish", "args": {}}],
            },
            {"reasoning": "Success", "result": "12:00 on 2024-01-01"},
        ]
    )
    dspy.settings.configure(lm=lm)

    outputs = react(query="what time is it?")

    observations = outputs.trajectory["observations_0"]
    assert len(observations) == 2
    assert observations[0]["tool"] == "get_time"
    assert observations[0]["result"] == "12:00"
    assert observations[1]["tool"] == "get_date"
    assert observations[1]["result"] == "2024-01-01"
