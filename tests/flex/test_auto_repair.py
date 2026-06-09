"""Tests for the auto-repair flow in dspy.Flex.

Covers both load-time (bind) failures and runtime failures from
user-edited code, plus the opt-out (``auto_repair=False``) path.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

import dspy
from dspy.flex import flex
from dspy.flex.exploration import FlexEvent
from dspy.utils.dummies import DummyLM

CANNED_PREDICTORS = textwrap.dedent("""
    PREDICTORS = {
        "echo": dspy.Predict("q -> a"),
    }
""").strip()

CANNED_FORWARD = textwrap.dedent("""
    def forward(self, q):
        out = self.echo(q=q)
        return dspy.Prediction(a=out.a)
""").strip()


def _swap_in_persisted(text: str, old: str, new: str, *, indent: str = "    ") -> str:
    return text.replace(textwrap.indent(old, indent), textwrap.indent(new, indent))


def _read_log(flex_root: Path, flex_id: str) -> list[dict]:
    log_path = flex_root / ".flex" / flex_id / "exploration.jsonl"
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


def _manifest_versions(flex_root: Path, flex_id: str) -> list[dict]:
    data = json.loads((flex_root / ".flex" / "manifest.json").read_text())
    return data["flex_modules"][flex_id]["versions"]


def _write_initial_flex_file(tmp_path: Path) -> Path:
    """Generate a clean Flex file on disk, return its path."""
    dspy.configure(lm=DummyLM([{"predictors_src": CANNED_PREDICTORS, "forward_src": CANNED_FORWARD}]))

    @flex(persist_to=str(tmp_path / "echo.py"), intent_check="off")
    class Echo(dspy.Signature):
        """Echo."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    Echo()  # writes the file
    return tmp_path / "echo.py"


def _make_echo_factory(persist_to: Path, *, auto_repair: bool = True):
    @flex(persist_to=str(persist_to), auto_repair=auto_repair, intent_check="off")
    class Echo(dspy.Signature):
        """Echo."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    return Echo


def test_load_time_repair_when_predictors_is_none(tmp_path: Path) -> None:
    """User clobbers PREDICTORS to None → bind raises → repair runs."""
    path = _write_initial_flex_file(tmp_path)

    # Break the persisted file: replace the predictors body with `PREDICTORS = None`.
    text = path.read_text()
    broken_text = _swap_in_persisted(text, CANNED_PREDICTORS, "PREDICTORS = None")
    assert "PREDICTORS = None" in broken_text  # sanity
    path.write_text(broken_text)

    # The repair LM returns the canned-good code.
    dspy.configure(lm=DummyLM([{"predictors_src": CANNED_PREDICTORS, "forward_src": CANNED_FORWARD}]))
    program = _make_echo_factory(path)()

    # File is back to a valid PREDICTORS dict.
    fixed = path.read_text()
    assert "PREDICTORS = None" not in fixed
    assert "PREDICTORS = {" in fixed

    # Forward works after the repair.
    assert program.forward_src is not None

    # Exploration log carries a REPAIR event followed by CODEGEN + ACCEPT.
    log = _read_log(tmp_path, "Echo")
    events = [e["event"] for e in log]
    assert FlexEvent.REPAIR.value in events
    assert FlexEvent.CODEGEN.value in events
    assert FlexEvent.ACCEPT.value in events

    # Manifest has 2 versions: the initial write + the repaired one.
    versions = _manifest_versions(tmp_path, "Echo")
    assert len(versions) >= 2
    assert "auto-repair" in versions[-1]["notes"]


def test_load_time_repair_off_surfaces_error(tmp_path: Path) -> None:
    """With auto_repair=False, a broken persisted file raises on construction."""
    path = _write_initial_flex_file(tmp_path)
    text = path.read_text()
    path.write_text(_swap_in_persisted(text, CANNED_PREDICTORS, "PREDICTORS = None"))

    dspy.configure(lm=DummyLM([{"predictors_src": CANNED_PREDICTORS, "forward_src": CANNED_FORWARD}]))
    factory = _make_echo_factory(path, auto_repair=False)
    with pytest.raises(RuntimeError, match="PREDICTORS"):
        factory()


def test_runtime_repair_on_attribute_error(tmp_path: Path) -> None:
    """User edits forward() to dereference None → runtime AttributeError → repair runs."""
    path = _write_initial_flex_file(tmp_path)

    # Break the forward body: shadow `out` with None, then access `out.a`.
    broken_forward = textwrap.dedent("""
        def forward(self, q):
            out = self.echo(q=q)
            out = None
            return dspy.Prediction(a=out.a)
    """).strip()
    text = path.read_text()
    path.write_text(_swap_in_persisted(text, CANNED_FORWARD, broken_forward))

    # The repair LM returns the canned-good forward.
    dspy.configure(
        lm=DummyLM(
            [
                # answer for the user's echo predictor call after the repair
                {"a": "world"},
                # repair codegen output
                {"predictors_src": CANNED_PREDICTORS, "forward_src": CANNED_FORWARD},
                # one more echo prediction in case the post-repair call re-queries
                {"a": "world"},
            ]
        )
    )
    program = _make_echo_factory(path)()
    # First call should auto-repair and then succeed on the re-run.
    result = program(q="hello")
    assert result.a == "world"

    # File now contains the fixed forward.
    fixed = path.read_text()
    assert "out = None" not in fixed

    # Exploration log has a REPAIR with failure_kind=runtime.
    log = _read_log(tmp_path, "Echo")
    repair_entries = [e for e in log if e["event"] == FlexEvent.REPAIR.value]
    assert any(e.get("failure_kind") == "runtime" for e in repair_entries)


def test_runtime_repair_runs_only_once(tmp_path: Path) -> None:
    """Even if the second call breaks again, Flex doesn't re-repair in the same process."""
    path = _write_initial_flex_file(tmp_path)

    # Break the forward body identically to above.
    broken_forward = textwrap.dedent("""
        def forward(self, q):
            out = self.echo(q=q)
            out = None
            return dspy.Prediction(a=out.a)
    """).strip()
    text = path.read_text()
    path.write_text(_swap_in_persisted(text, CANNED_FORWARD, broken_forward))

    # Repair LM returns the *same* broken forward, so the second call still raises
    # — and Flex should propagate without attempting another repair.
    dspy.configure(
        lm=DummyLM(
            [
                {"a": "anything"},  # first call: echo before the None deref
                {"predictors_src": CANNED_PREDICTORS, "forward_src": broken_forward},  # repair
                {"a": "anything"},  # post-repair re-run: echo before the (still-broken) None deref
                {"a": "anything"},  # second call: echo before the None deref (no second repair)
            ]
        )
    )
    program = _make_echo_factory(path)()

    with pytest.raises(AttributeError):
        program(q="hello")

    # And a second call also raises directly (no second repair).
    with pytest.raises(AttributeError):
        program(q="hello")

    log = _read_log(tmp_path, "Echo")
    runtime_repairs = [
        e for e in log if e["event"] == FlexEvent.REPAIR.value and e.get("failure_kind") == "runtime"
    ]
    assert len(runtime_repairs) == 1


def test_runtime_does_not_repair_non_user_errors(tmp_path: Path) -> None:
    """RuntimeError raised inside forward() bypasses auto-repair."""
    path = _write_initial_flex_file(tmp_path)
    # Break the forward body with a raise of a non-repairable error class.
    broken_forward = textwrap.dedent("""
        def forward(self, q):
            raise RuntimeError("boom from downstream")
    """).strip()
    text = path.read_text()
    path.write_text(_swap_in_persisted(text, CANNED_FORWARD, broken_forward))

    # Repair LM is configured but should never be called.
    dspy.configure(lm=DummyLM([{"predictors_src": CANNED_PREDICTORS, "forward_src": CANNED_FORWARD}]))
    program = _make_echo_factory(path)()

    with pytest.raises(RuntimeError, match="boom"):
        program(q="hello")

    log = _read_log(tmp_path, "Echo")
    assert not any(
        e["event"] == FlexEvent.REPAIR.value and e.get("failure_kind") == "runtime" for e in log
    )


def test_runtime_repair_off_surfaces_error(tmp_path: Path) -> None:
    """auto_repair=False propagates runtime errors from forward() directly."""
    path = _write_initial_flex_file(tmp_path)
    broken_forward = textwrap.dedent("""
        def forward(self, q):
            out = self.echo(q=q)
            out = None
            return dspy.Prediction(a=out.a)
    """).strip()
    text = path.read_text()
    path.write_text(_swap_in_persisted(text, CANNED_FORWARD, broken_forward))

    dspy.configure(lm=DummyLM([{"a": "x"}]))
    factory = _make_echo_factory(path, auto_repair=False)
    program = factory()
    with pytest.raises(AttributeError):
        program(q="hello")
