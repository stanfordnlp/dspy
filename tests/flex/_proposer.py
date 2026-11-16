"""Turns of a scripted reflection model as GEPA's code proposer, a ``dspy.ReAct`` that writes and edits the program's
files and runs its tests."""

import dspy
from dspy.predict.flex import program


def proposes(revision: str) -> list[dict]:
    """The turns that write the files of ``revision`` (``program.read``), run the tests and submit the program."""
    return [runs_tests(revision), submits()]


def runs_tests(revision: str = "") -> dict:
    """The turn that writes the files of ``revision`` and then runs the tests."""
    writes = [{"name": "write", "args": {"file": name, "source": source}} for name, source in program.read(revision).items()]
    return _turn(*writes, {"name": "run_tests", "args": {}})


def edits(file: str, old: str, new: str) -> dict:
    """The turn that edits ``file`` and then runs the tests."""
    return _turn({"name": "edit", "args": {"file": file, "old": old, "new": new}}, {"name": "run_tests", "args": {}})


def submits() -> dict:
    """The turn that submits the program as it is."""
    return _turn({"name": "submit", "args": {}})


def _turn(*calls: dict) -> dict:
    return {"next_thought": "Next.", "tool_calls": dspy.ToolCalls.from_dict_list(list(calls))}
