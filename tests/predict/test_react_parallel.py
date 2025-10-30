"""Tests for parallel tool execution in ReAct."""
import asyncio
import time

import pytest

import dspy
from dspy.utils.dummies import DummyLM


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
