"""The dspy.Flex baseline is deterministic and LM-free.

Constructing a Flex for a given signature binds the same ``dspy.RLM`` baseline
source every time, with no LM call — two constructions of the same signature
produce byte-identical ``module_src``, and the baseline survives the persistence
round-trip.
"""

from __future__ import annotations

from pathlib import Path

import dspy
from dspy.flex import Flex


class Doubler(dspy.Signature):
    """Return double the input value."""

    value: int = dspy.InputField()
    result: int = dspy.OutputField()


def test_baseline_is_lm_free_and_deterministic(tmp_path: Path) -> None:
    # No LM configured anywhere: construction must not make any LM call.
    a = Flex(Doubler, persist_to=str(tmp_path / "a.py"))
    b = Flex(Doubler, persist_to=str(tmp_path / "b.py"))

    # Byte-identical baseline source for the same signature.
    assert a.module_src == b.module_src

    # It's the typed RLM baseline (a dspy.Module subclass) that unwraps the declared output.
    assert "class DoublerModule(dspy.Module)" in a.module_src
    assert "dspy.RLM(" in a.module_src
    assert "value: int -> result: int" in a.module_src
    assert "result.result" in a.module_src


def test_baseline_roundtrips_through_disk(tmp_path: Path) -> None:
    persist_path = tmp_path / "doubler.py"
    program = Flex(Doubler, persist_to=str(persist_path))

    text = persist_path.read_text()
    assert "dspy.RLM(" in text
    assert "def forward" in text
    assert "class DoublerModule(dspy.Module)" in text

    # Reconstruct with no LM at all — must load the baseline body from disk and
    # bind the same source (the RLM string signature is preserved).
    program2 = Flex(Doubler, persist_to=str(persist_path))
    assert "dspy.RLM(" in program2.module_src
    assert "value: int -> result: int" in program2.module_src
    assert program2.module_src.strip() == program.module_src.strip()
