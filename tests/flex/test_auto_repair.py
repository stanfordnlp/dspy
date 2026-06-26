"""Tests for the auto-repair flow in dspy.Flex.

Covers load/bind-time failures and runtime failures from edited code, plus the opt-out
(``auto_repair=False``) path.

Construction is LM-free (it binds the deterministic RLM baseline). To exercise repair we bind
a *plain dspy.Predict* module class in memory (via ``_bind_code`` / ``_bind_with_repair``) and
break it, with a repair DummyLM that returns the good plain-Predict class — never running the
heavy RLM baseline.
"""

from __future__ import annotations

import pytest

import dspy
from dspy.flex import Flex
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
BROKEN_MODULE = CANNED_MODULE.replace(GOOD_FORWARD, BROKEN_FORWARD)
# Not a dspy.Module subclass at all -> _bind_code raises (a bind-time failure).
UNBINDABLE = "class EchoModule:\n    pass"


class Echo(dspy.Signature):
    """Echo."""

    q: str = dspy.InputField()
    a: str = dspy.OutputField()


def _flex_with(module_src: str, *, auto_repair: bool = True) -> Flex:
    """A Flex whose code is `module_src`, bound in memory (LM-free)."""
    flex = Flex(Echo, auto_repair=auto_repair)
    flex._bind_code(module_src)
    return flex


def test_bind_time_repair_on_bind_failure() -> None:
    """A loaded/edited body that fails to bind is auto-repaired (this is the load-path repair)."""
    program = Flex(Echo)
    dspy.configure(lm=DummyLM([{"module_src": CANNED_MODULE}]))
    program._bind_with_repair(UNBINDABLE)  # bind fails -> repair runs

    assert "class EchoModule(dspy.Module)" in program.module_src
    assert program.module_src is not None


def test_bind_time_repair_off_surfaces_error() -> None:
    """With auto_repair=False, a body that fails to bind raises directly."""
    program = Flex(Echo, auto_repair=False)
    dspy.configure(lm=DummyLM([{"module_src": CANNED_MODULE}]))
    with pytest.raises(RuntimeError, match="Module subclass"):
        program._bind_with_repair(UNBINDABLE)


def test_runtime_repair_on_attribute_error() -> None:
    """Edited forward() dereferences None -> runtime AttributeError -> repair runs."""
    dspy.configure(
        lm=DummyLM(
            [
                {"a": "world"},  # echo call before the None deref
                {"module_src": CANNED_MODULE},  # repair codegen
                {"a": "world"},  # post-repair re-run
            ]
        )
    )
    program = _flex_with(BROKEN_MODULE)
    # First call should auto-repair and then succeed on the re-run.
    result = program(q="hello")
    assert result.a == "world"

    # The bound code now contains the fixed forward (in memory; persist with save() to keep it).
    assert "out = None" not in program.module_src


def test_runtime_repair_runs_only_once() -> None:
    """Even if repair returns broken code again, Flex doesn't re-repair in the same process."""
    # Repair LM returns the *same* broken module, so the post-repair re-run still raises
    # — Flex must propagate without attempting another repair on the next call.
    dspy.configure(
        lm=DummyLM(
            [
                {"a": "anything"},
                {"module_src": BROKEN_MODULE},
                {"a": "anything"},
                {"a": "anything"},
            ]
        )
    )
    program = _flex_with(BROKEN_MODULE)

    with pytest.raises(AttributeError):
        program(q="hello")
    # A second call raises directly (the one-shot repair was already used).
    with pytest.raises(AttributeError):
        program(q="hello")


def test_runtime_does_not_repair_non_user_errors() -> None:
    """A RuntimeError raised inside forward() bypasses auto-repair and propagates."""
    raising_module = CANNED_MODULE.replace(
        GOOD_FORWARD, '    def forward(self, q):\n        raise RuntimeError("boom from downstream")'
    )
    dspy.configure(lm=DummyLM([{"module_src": CANNED_MODULE}]))
    program = _flex_with(raising_module)

    with pytest.raises(RuntimeError, match="boom"):
        program(q="hello")
    # The broken forward is unchanged (no repair was attempted).
    assert "boom from downstream" in program.module_src


def test_runtime_repair_off_surfaces_error() -> None:
    """auto_repair=False propagates runtime errors from forward() directly."""
    dspy.configure(lm=DummyLM([{"a": "x"}]))
    program = _flex_with(BROKEN_MODULE, auto_repair=False)
    with pytest.raises(AttributeError):
        program(q="hello")
