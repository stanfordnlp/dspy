"""Turns of a scripted reflection model as GEPA's code proposer, a ``dspy.ReAct`` with a ``run_tests`` tool."""

import dspy


def proposes(revision: str) -> list[dict]:
    """The turns that test ``revision`` and submit the program as tested."""
    return [runs_tests(revision), submits()]


def runs_tests(revision: str) -> dict:
    """The turn that applies ``revision`` to the program and runs its tests."""
    return _calls("run_tests", revision=revision)


def submits() -> dict:
    """The turn that submits the program as last tested."""
    return _calls("submit")


def _calls(tool: str, **args) -> dict:
    return {"next_thought": "Next.", "tool_calls": dspy.ToolCalls.from_dict_list([{"name": tool, "args": args}])}
