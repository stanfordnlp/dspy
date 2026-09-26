"""Filesystem-backed nested RLMs, including the Flex/GEPA source adaptation path."""

from functools import partial
from pathlib import PurePosixPath

import pytest

import dspy
from dspy.utils.dummies import DummyLM
from tests.predict.test_rlm import make_mock_predictor

monty = pytest.importorskip("pydantic_monty")


def test_nested_rlm_shares_explicit_filesystem_but_not_python_globals():
    fs = monty.OSAccess([monty.MemoryFile("/work/input.txt", "13\n29")])
    factory = partial(dspy.MontyInterpreter, os=fs, cwd="/work")
    rlm = dspy.RLM("query -> answer: int", interpreter_factory=factory, max_iters=2)
    rlm.generate_action = make_mock_predictor([
        {"reasoning": "Delegate", "code": (
            "sentinel = 17\n"
            "child = dspy.RLM('query -> answer: int', max_iters=1)\n"
            "print(child(query=query).answer)"
        )},
        {"reasoning": "Read artifact", "code": (
            "from pathlib import Path\nSUBMIT(answer=int(Path('result.txt').read_text()) + sentinel)"
        )},
    ])
    lm = DummyLM([{"reasoning": "Read and aggregate", "code": (
        "from pathlib import Path\n"
        "sentinel = 999\n"
        "total = sum([int(line) for line in Path('input.txt').read_text().splitlines()])\n"
        "Path('result.txt').write_text(str(total))\n"
        "SUBMIT(answer=total)"
    )}])
    with dspy.context(lm=lm):
        result = rlm(query="Sum the file")
    assert result.answer == 59
    assert result.trajectory[0]["output"] == "42"
    assert fs.path_read_text(PurePosixPath("/work/result.txt")) == "42"


def test_gepa_rebound_flex_runs_native_file_operations():
    from dspy.teleprompt.gepa.gepa_utils import DspyAdapter

    fs = monty.OSAccess([monty.MemoryFile("/work/input.txt", "13\n29")])
    factory = partial(dspy.MontyInterpreter, os=fs, cwd="/work")
    student = dspy.Flex("path -> answer: int", interpreter_factory=factory)
    adapter = DspyAdapter(student_module=student, metric_fn=lambda gold, pred: float(pred.answer == 42), feedback_map={})
    source = (
        "from pathlib import Path\n"
        "class Reader(dspy.Module):\n"
        "    def forward(self, path):\n"
        "        with open(path) as stream:\n"
        "            numbers = [int(line) for line in stream.read().splitlines()]\n"
        "        answer = sum(numbers)\n"
        "        Path('result.txt').write_text(str(answer))\n"
        "        return dspy.Prediction(answer=int(Path('result.txt').read_text()))"
    )
    rebuilt = adapter.build_program({"self": source})
    assert rebuilt(path="input.txt").answer == 42
    assert rebuilt.module_src == source
    # GEPA deep-copies this partial factory and its in-memory filesystem.
    assert not fs.path_exists(PurePosixPath("/work/result.txt"))
