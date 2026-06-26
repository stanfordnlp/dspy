"""The dspy.Vibe baseline is deterministic and LM-free.

Constructing a Vibe for a given signature binds the same ``dspy.RLM`` baseline
source every time, with no LM call — two constructions of the same signature
produce byte-identical ``predictors_src`` / ``forward_src``, and the baseline
survives the persistence round-trip.
"""

from __future__ import annotations

from pathlib import Path

import dspy
from dspy.vibe import Vibe


class Doubler(dspy.Signature):
    """Return double the input value."""

    value: int = dspy.InputField()
    result: int = dspy.OutputField()


def test_baseline_is_lm_free_and_deterministic(tmp_path: Path) -> None:
    # No LM configured anywhere: construction must not make any LM call.
    a = Vibe(Doubler, persist_to=str(tmp_path / "a.py"))
    b = Vibe(Doubler, persist_to=str(tmp_path / "b.py"))

    # Byte-identical baseline source for the same signature.
    assert a.predictors_src == b.predictors_src
    assert a.forward_src == b.forward_src

    # It's the typed RLM baseline that unwraps the declared output.
    assert "dspy.RLM(" in a.predictors_src
    assert "value: int -> result: int" in a.predictors_src
    assert "result.result" in a.forward_src


def test_baseline_roundtrips_through_disk(tmp_path: Path) -> None:
    persist_path = tmp_path / "doubler.py"
    program = Vibe(Doubler, persist_to=str(persist_path))

    text = persist_path.read_text()
    assert "dspy.RLM(" in text
    assert "def forward" in text

    # Reconstruct with no LM at all — must load the baseline body from disk and
    # bind the same source (the RLM string signature is preserved).
    program2 = Vibe(Doubler, persist_to=str(persist_path))
    assert "dspy.RLM(" in program2.predictors_src
    assert "value: int -> result: int" in program2.predictors_src
    assert program2.forward_src.strip() == program.forward_src.strip()
