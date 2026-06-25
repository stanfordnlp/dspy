"""Tests for the continuous-improvement loop: dataset-backed Flex.improve().

Uses lightweight stand-in "improvers" (duck-typed on
``compile(student, *, trainset, valset)``) instead of a real dspy.GEPA, so the
tests stay fast and deterministic. The real optimizer is exercised by
tests/flex/demo/*.
"""

from __future__ import annotations

import textwrap
import types
from pathlib import Path

import pytest

import dspy
from dspy.flex import Flex
from dspy.utils.dummies import DummyLM

NEW_PREDICTORS = textwrap.dedent("""
    PREDICTORS = {
        "echo2": dspy.Predict("q -> a"),
    }
""").strip()

NEW_FORWARD = textwrap.dedent("""
    def forward(self, q):
        out = self.echo2(q=q)
        return dspy.Prediction(a=out.a)
""").strip()

TRAIN = [dspy.Example(q="x", a="y").with_inputs("q")]


class Echo(dspy.Signature):
    """Echo."""

    q: str = dspy.InputField()
    a: str = dspy.OutputField()


def _echo_factory(persist_to: Path | None, *, improver=None):
    def factory():
        return Flex(
            Echo,
            persist_to=str(persist_to) if persist_to else None,
            improver=improver,
        )

    return factory


def _write_initial(tmp_path: Path) -> Path:
    path = tmp_path / "echo.py"
    _echo_factory(path)()  # LM-free baseline write
    # The baseline's in-memory forward_src carries a trailing newline that the
    # persistence round-trip strips, so the first reload backfills a matching
    # body hash. Do that reload here so the on-disk file is pristine and a
    # subsequent construction is not spuriously flagged as a manual edit.
    dspy.configure(lm=DummyLM([]))
    _echo_factory(path)()
    return path


def _simulate_manual_edit(path: Path) -> None:
    # Edit the RLM-baseline body: unwrap-and-uppercase the declared output.
    text = path.read_text()
    edited = text.replace("result.a)", "result.a.upper())")
    assert edited != text, "manual-edit simulation did not change the file"
    path.write_text(edited)


class _SpyImprover:
    """Records calls and returns a fixed new implementation (no disk writes)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def compile(self, student, *, trainset, valset):
        self.calls.append({"student": student, "trainset": trainset, "valset": valset})
        return types.SimpleNamespace(predictors_src=NEW_PREDICTORS, forward_src=NEW_FORWARD)


class _RewritingImprover:
    """Mimics dspy.GEPA's externally-visible effect: writes the new body to disk."""

    def __init__(self) -> None:
        self.calls: list = []

    def compile(self, student, *, trainset, valset):
        self.calls.append(student)
        student._write_persisted(NEW_PREDICTORS, NEW_FORWARD, student._signature_hash())
        return types.SimpleNamespace(predictors_src=NEW_PREDICTORS, forward_src=NEW_FORWARD)


# --- guard / no-op paths -----------------------------------------------------


def test_improve_without_improver_raises(tmp_path: Path) -> None:
    path = _write_initial(tmp_path)
    dspy.configure(lm=DummyLM([]))
    program = _echo_factory(path)()
    with pytest.raises(ValueError, match="improver"):
        program.improve()


def test_improve_is_noop_without_a_pending_edit(tmp_path: Path) -> None:
    path = _write_initial(tmp_path)
    spy = _SpyImprover()
    dspy.configure(lm=DummyLM([]))
    program = _echo_factory(path, improver=spy)()  # clean reload, no edit
    assert program.improve() is program
    assert spy.calls == []


def test_improve_without_dataset_raises(tmp_path: Path) -> None:
    path = _write_initial(tmp_path)
    _simulate_manual_edit(path)
    spy = _SpyImprover()
    dspy.configure(lm=DummyLM([]))
    program = _echo_factory(path, improver=spy)()  # edit detected, but nothing saved
    with pytest.raises(ValueError, match="dataset"):
        program.improve()
    assert spy.calls == []


def test_improve_in_memory_raises(tmp_path: Path) -> None:
    program = _echo_factory(None, improver=_SpyImprover())()  # persist_to=None
    with pytest.raises(ValueError, match="in-memory"):
        program.improve(force=True)


# --- active paths ------------------------------------------------------------


def test_improve_runs_on_a_pending_edit(tmp_path: Path) -> None:
    path = _write_initial(tmp_path)
    _simulate_manual_edit(path)
    spy = _SpyImprover()
    dspy.configure(lm=DummyLM([]))
    program = _echo_factory(path, improver=spy)()
    assert program._pending_manual_edit is True
    auto_repair_before = program._auto_repair

    out = program.improve(trainset=TRAIN)

    assert out is program
    assert len(spy.calls) == 1
    assert spy.calls[0]["student"] is program  # seeded from the edited module itself
    assert program.forward_src == NEW_FORWARD  # rebound to the improved code
    assert "echo2" in program.predictors_src
    assert program._pending_manual_edit is False  # flag consumed
    assert program._auto_repair == auto_repair_before  # improve() leaves it alone
    assert program._improver is spy  # improver restored after the run


def test_improve_force_runs_without_an_edit(tmp_path: Path) -> None:
    path = _write_initial(tmp_path)
    spy = _SpyImprover()
    dspy.configure(lm=DummyLM([]))
    program = _echo_factory(path, improver=spy)()  # no edit
    program.improve(trainset=TRAIN, force=True)
    assert len(spy.calls) == 1


def test_set_improver_is_chainable_and_used(tmp_path: Path) -> None:
    path = _write_initial(tmp_path)
    _simulate_manual_edit(path)
    dspy.configure(lm=DummyLM([]))
    program = _echo_factory(path)()  # no improver at construction
    spy = _SpyImprover()
    assert program.set_improver(spy) is program  # chainable
    program.improve(trainset=TRAIN)
    assert len(spy.calls) == 1


def test_improve_rewrites_file_so_next_run_sees_no_edit(tmp_path: Path) -> None:
    """Loop-safety: after improve() rewrites the file, a fresh reconstruct is pristine."""
    path = _write_initial(tmp_path)
    _simulate_manual_edit(path)
    rewriter = _RewritingImprover()
    dspy.configure(lm=DummyLM([]))
    program = _echo_factory(path, improver=rewriter)()
    assert program._pending_manual_edit is True

    program.improve(trainset=TRAIN)
    assert len(rewriter.calls) == 1
    assert program.forward_src == NEW_FORWARD

    program2 = _echo_factory(path)()  # fresh process would do this
    assert program2._pending_manual_edit is False
    assert program2.forward_src == NEW_FORWARD


# --- improver wiring (constructor) -------------------------------------------


def test_constructor_accepts_improver(tmp_path: Path) -> None:
    spy = _SpyImprover()
    program = Flex(Echo, persist_to=str(tmp_path / "e.py"), improver=spy)
    assert program._improver is spy
