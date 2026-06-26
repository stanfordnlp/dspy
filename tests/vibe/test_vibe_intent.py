"""Intent detection: dspy.Vibe warns at construction when a signature looks vague.

The check is best-effort and runs one LM call only on a fresh generation (new/changed
signature or in-memory mode), never on a plain reload, and only when an LM is available.
A vague verdict produces a warning naming what's unclear plus a clarifying question; the
module is still built.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import dspy
from dspy.utils.dummies import DummyLM
from dspy.vibe import Vibe


class Vague(dspy.Signature):
    """Do the thing."""

    data: str = dspy.InputField()
    result: str = dspy.OutputField()


def _vague_lm(aspect: str = "ASPECT", question: str = "QUESTION") -> DummyLM:
    return DummyLM([{"is_clear": False, "vague_aspect": aspect, "clarifying_question": question}])


def _clear_lm() -> DummyLM:
    return DummyLM([{"is_clear": True, "vague_aspect": "", "clarifying_question": ""}])


def _vague_warnings(records) -> list[str]:
    return [str(w.message) for w in records if "vague or misleading" in str(w.message)]


def test_warns_when_signature_is_vague() -> None:
    dspy.configure(lm=_vague_lm())
    # In-memory construction is a fresh generation, so the intent check runs.
    with pytest.warns(UserWarning, match="vague or misleading"):
        Vibe(Vague)


def test_warning_names_the_vague_aspect_and_clarifying_question() -> None:
    dspy.configure(lm=_vague_lm(aspect="THE_DATA_FIELD_IS_UNTYPED", question="WHAT_TASK_IS_THIS"))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Vibe(Vague)
    msgs = _vague_warnings(rec)
    assert len(msgs) == 1, f"expected exactly one intent warning, got {msgs}"
    assert "THE_DATA_FIELD_IS_UNTYPED" in msgs[0]  # what is vague
    assert "WHAT_TASK_IS_THIS" in msgs[0]  # the clarifying question


def test_no_warning_when_signature_is_clear() -> None:
    dspy.configure(lm=_clear_lm())
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        program = Vibe(Vague)
    assert _vague_warnings(rec) == []
    assert program.module_src is not None


def test_skipped_when_no_lm_configured() -> None:
    with dspy.context(lm=None), warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        program = Vibe(Vague)  # no LM -> intent check skipped, construction stays LM-free
    assert _vague_warnings(rec) == []
    assert program.module_src is not None


def test_check_intent_false_disables_the_call() -> None:
    dspy.configure(lm=_vague_lm())  # would warn if the check ran
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        Vibe(Vague, check_intent=False)
    assert _vague_warnings(rec) == []


def test_not_rerun_on_plain_reload(tmp_path: Path) -> None:
    path = tmp_path / "vague.py"
    # Fresh generation: warns and consumes the single DummyLM response.
    dspy.configure(lm=_vague_lm())
    with pytest.warns(UserWarning, match="vague or misleading"):
        Vibe(Vague, persist_to=str(path))

    # Reload of the matching persisted file is the load path: no intent call, no warning.
    # (The DummyLM is now empty; if the check ran it would error, but it's skipped.)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        reloaded = Vibe(Vague, persist_to=str(path))
    assert _vague_warnings(rec) == []
    assert reloaded.module_src is not None
