"""Turns of a scripted reflection model as GEPA's code proposer, a ``dspy.ReAct`` with a ``run_tests`` tool."""

import dspy


def submits(source: str) -> dict:
    """The turn that submits ``source`` as the revision."""
    return _calls("submit", revised_source=source)


def runs_tests(revision: str) -> dict:
    """The turn that runs the tests of the program as ``revision`` would make it."""
    return _calls("run_tests", revision=revision)


def _calls(tool: str, **args) -> dict:
    return {"next_thought": "Next.", "tool_calls": dspy.ToolCalls.from_dict_list([{"name": tool, "args": args}])}
