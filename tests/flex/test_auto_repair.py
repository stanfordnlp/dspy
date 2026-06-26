"""Tests for the auto-repair flow in dspy.Flex.

Covers load-time (bind) failures and runtime failures from user-edited code, plus the
opt-out (``auto_repair=False``) path.

Construction is LM-free (it binds the deterministic RLM baseline). To exercise repair we
persist a *plain dspy.Predict* module class, break it, and reload with a repair DummyLM that
returns the good plain-Predict class — never running the heavy RLM baseline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import dspy
from dspy.flex import Flex
from dspy.flex.persistence import parse_persisted_file, render_persisted_file
from dspy.utils.dummies import DummyLM

# The good forward method, exactly as it appears (4-space indented) in CANNED_MODULE.
GOOD_FORWARD = "    def forward(self, q):\n        out = self.echo(q=q)\n        return dspy.Prediction(a=out.a)"
# A runtime-broken forward: dereferences None -> AttributeError when called.
BROKEN_FORWARD = (
    "    def forward(self, q):\n"
    "        out = self.echo(q=q)\n"
    "        out = None\n"
    "        return dspy.Prediction(a=out.a)"
)
CANNED_MODULE = (
    "class EchoModule(dspy.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    '        self.echo = dspy.Predict("q -> a")\n'
    "\n" + GOOD_FORWARD
)


class Echo(dspy.Signature):
    """Echo."""

    q: str = dspy.InputField()
    a: str = dspy.OutputField()


def _write_initial_flex_file(tmp_path: Path) -> Path:
    """Construct a Flex (LM-free baseline) and rewrite its body to a plain dspy.Predict module
    class, keeping the signature hash intact so it loads as a runnable (non-RLM) module. Returns
    the persisted file path."""
    path = tmp_path / "echo.py"
    Flex(Echo, persist_to=str(path))  # writes the RLM baseline file

    parsed = parse_persisted_file(path.read_text(encoding="utf-8"))
    assert parsed is not None
    path.write_text(
        render_persisted_file(
            signature_hash=parsed.signature_hash,
            signature_name="Echo",
            module_src=CANNED_MODULE,
            signature_spec="q: str -> a: str",
        ),
        encoding="utf-8",
    )
    return path


def _make_echo_factory(persist_to: Path, *, auto_repair: bool = True):
    def factory():
        return Flex(Echo, persist_to=str(persist_to), auto_repair=auto_repair)

    return factory


def test_load_time_repair_on_bind_failure(tmp_path: Path) -> None:
    """User breaks the class so it no longer subclasses dspy.Module -> bind raises -> repair runs."""
    path = _write_initial_flex_file(tmp_path)

    text = path.read_text(encoding="utf-8")
    broken_text = text.replace("class EchoModule(dspy.Module):", "class EchoModule:")
    assert "class EchoModule:" in broken_text  # sanity
    path.write_text(broken_text, encoding="utf-8")

    # The repair LM returns the canned-good class.
    dspy.configure(lm=DummyLM([{"module_src": CANNED_MODULE}]))
    program = _make_echo_factory(path)()

    # File is back to a valid dspy.Module subclass and the module bound successfully.
    fixed = path.read_text(encoding="utf-8")
    assert "class EchoModule:" not in fixed
    assert "class EchoModule(dspy.Module)" in fixed
    assert program.module_src is not None


def test_load_time_repair_off_surfaces_error(tmp_path: Path) -> None:
    """With auto_repair=False, a broken persisted file raises on construction."""
    path = _write_initial_flex_file(tmp_path)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("class EchoModule(dspy.Module):", "class EchoModule:"), encoding="utf-8")

    dspy.configure(lm=DummyLM([{"module_src": CANNED_MODULE}]))
    factory = _make_echo_factory(path, auto_repair=False)
    with pytest.raises(RuntimeError, match="Module subclass"):
        factory()


def test_runtime_repair_on_attribute_error(tmp_path: Path) -> None:
    """User edits forward() to dereference None -> runtime AttributeError -> repair runs."""
    path = _write_initial_flex_file(tmp_path)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(GOOD_FORWARD, BROKEN_FORWARD), encoding="utf-8")

    dspy.configure(
        lm=DummyLM(
            [
                {"a": "world"},  # echo call before the None deref
                {"module_src": CANNED_MODULE},  # repair codegen
                {"a": "world"},  # post-repair re-run
            ]
        )
    )
    program = _make_echo_factory(path)()
    # First call should auto-repair and then succeed on the re-run.
    result = program(q="hello")
    assert result.a == "world"

    # File now contains the fixed forward.
    assert "out = None" not in path.read_text(encoding="utf-8")


def test_runtime_repair_runs_only_once(tmp_path: Path) -> None:
    """Even if repair returns broken code again, Flex doesn't re-repair in the same process."""
    path = _write_initial_flex_file(tmp_path)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(GOOD_FORWARD, BROKEN_FORWARD), encoding="utf-8")

    broken_module = CANNED_MODULE.replace(GOOD_FORWARD, BROKEN_FORWARD)
    # Repair LM returns the *same* broken module, so the post-repair re-run still raises
    # — Flex must propagate without attempting another repair on the next call.
    dspy.configure(
        lm=DummyLM(
            [
                {"a": "anything"},
                {"module_src": broken_module},
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
    raising_forward = '    def forward(self, q):\n        raise RuntimeError("boom from downstream")'
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(GOOD_FORWARD, raising_forward), encoding="utf-8")

    dspy.configure(lm=DummyLM([{"module_src": CANNED_MODULE}]))
    program = _make_echo_factory(path)()

    with pytest.raises(RuntimeError, match="boom"):
        program(q="hello")
    # The broken forward is unchanged (no repair was attempted).
    assert "boom from downstream" in path.read_text(encoding="utf-8")


def test_runtime_repair_off_surfaces_error(tmp_path: Path) -> None:
    """auto_repair=False propagates runtime errors from forward() directly."""
    path = _write_initial_flex_file(tmp_path)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(GOOD_FORWARD, BROKEN_FORWARD), encoding="utf-8")

    dspy.configure(lm=DummyLM([{"a": "x"}]))
    factory = _make_echo_factory(path, auto_repair=False)
    program = factory()
    with pytest.raises(AttributeError):
        program(q="hello")
