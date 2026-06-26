"""Tests for the auto-repair flow in dspy.Flex.

Covers load-time (bind) failures and runtime failures from user-edited code, plus the
opt-out (``auto_repair=False``) path.

Construction is LM-free (it binds the deterministic RLM baseline). To exercise repair we
persist a *plain dspy.Predict* body, break it, and reload with a repair DummyLM that
returns the good plain-Predict body — never running the heavy RLM baseline.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import dspy
from dspy.utils.dummies import DummyLM
from dspy.vibe import Flex
from dspy.vibe.persistence import parse_persisted_file, render_persisted_file

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


class Echo(dspy.Signature):
    """Echo."""

    q: str = dspy.InputField()
    a: str = dspy.OutputField()


def _swap_in_persisted(text: str, old: str, new: str, *, indent: str = "    ") -> str:
    return text.replace(textwrap.indent(old, indent), textwrap.indent(new, indent))


def _write_initial_flex_file(tmp_path: Path) -> Path:
    """Construct a Flex (LM-free baseline) and rewrite its body to a plain dspy.Predict
    implementation, keeping the signature hash intact so it loads as a runnable (non-RLM)
    body. Returns the persisted file path."""
    path = tmp_path / "echo.py"
    Flex(Echo, persist_to=str(path))  # writes the RLM baseline file

    parsed = parse_persisted_file(path.read_text())
    assert parsed is not None
    path.write_text(
        render_persisted_file(
            signature_hash=parsed.signature_hash,
            signature_name="Echo",
            predictors_src=CANNED_PREDICTORS,
            forward_src=CANNED_FORWARD,
        )
    )
    return path


def _make_echo_factory(persist_to: Path, *, auto_repair: bool = True):
    def factory():
        return Flex(Echo, persist_to=str(persist_to), auto_repair=auto_repair)

    return factory


def test_load_time_repair_when_predictors_is_none(tmp_path: Path) -> None:
    """User clobbers PREDICTORS to None → bind raises → repair runs and rewrites the file."""
    path = _write_initial_flex_file(tmp_path)

    text = path.read_text()
    broken_text = _swap_in_persisted(text, CANNED_PREDICTORS, "PREDICTORS = None")
    assert "PREDICTORS = None" in broken_text  # sanity
    path.write_text(broken_text)

    # The repair LM returns the canned-good code.
    dspy.configure(lm=DummyLM([{"predictors_src": CANNED_PREDICTORS, "forward_src": CANNED_FORWARD}]))
    program = _make_echo_factory(path)()

    # File is back to a valid PREDICTORS dict and the module bound successfully.
    fixed = path.read_text()
    assert "PREDICTORS = None" not in fixed
    assert "PREDICTORS = {" in fixed
    assert program.forward_src is not None


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

    broken_forward = textwrap.dedent("""
        def forward(self, q):
            out = self.echo(q=q)
            out = None
            return dspy.Prediction(a=out.a)
    """).strip()
    text = path.read_text()
    path.write_text(_swap_in_persisted(text, CANNED_FORWARD, broken_forward))

    dspy.configure(
        lm=DummyLM(
            [
                {"a": "world"},  # echo call before the None deref
                {"predictors_src": CANNED_PREDICTORS, "forward_src": CANNED_FORWARD},  # repair codegen
                {"a": "world"},  # post-repair re-run
            ]
        )
    )
    program = _make_echo_factory(path)()
    # First call should auto-repair and then succeed on the re-run.
    result = program(q="hello")
    assert result.a == "world"

    # File now contains the fixed forward.
    assert "out = None" not in path.read_text()


def test_runtime_repair_runs_only_once(tmp_path: Path) -> None:
    """Even if repair returns broken code again, Flex doesn't re-repair in the same process."""
    path = _write_initial_flex_file(tmp_path)

    broken_forward = textwrap.dedent("""
        def forward(self, q):
            out = self.echo(q=q)
            out = None
            return dspy.Prediction(a=out.a)
    """).strip()
    text = path.read_text()
    path.write_text(_swap_in_persisted(text, CANNED_FORWARD, broken_forward))

    # Repair LM returns the *same* broken forward, so the post-repair re-run still raises
    # — Flex must propagate without attempting another repair on the next call.
    dspy.configure(
        lm=DummyLM(
            [
                {"a": "anything"},
                {"predictors_src": CANNED_PREDICTORS, "forward_src": broken_forward},
                {"a": "anything"},
                {"a": "anything"},
            ]
        )
    )
    program = _make_echo_factory(path)()

    with pytest.raises(AttributeError):
        program(q="hello")
    # A second call raises directly (the one-shot repair was already used).
    with pytest.raises(AttributeError):
        program(q="hello")


def test_runtime_does_not_repair_non_user_errors(tmp_path: Path) -> None:
    """A RuntimeError raised inside forward() bypasses auto-repair and propagates."""
    path = _write_initial_flex_file(tmp_path)
    broken_forward = textwrap.dedent("""
        def forward(self, q):
            raise RuntimeError("boom from downstream")
    """).strip()
    text = path.read_text()
    path.write_text(_swap_in_persisted(text, CANNED_FORWARD, broken_forward))

    dspy.configure(lm=DummyLM([{"predictors_src": CANNED_PREDICTORS, "forward_src": CANNED_FORWARD}]))
    program = _make_echo_factory(path)()

    with pytest.raises(RuntimeError, match="boom"):
        program(q="hello")
    # The broken forward is unchanged (no repair was attempted).
    assert "boom from downstream" in path.read_text()


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
