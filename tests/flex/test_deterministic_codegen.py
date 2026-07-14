"""The dspy.Flex baseline is deterministic and LM-free.

Constructing a Flex for a given signature binds the same baseline source every time,
with no LM call — two constructions of the same signature produce byte-identical
``module_src``, and the baseline survives a save/load round-trip (Module.save / load
carries the code). With no tools, the baseline is a single ``dspy.Predict``.
"""

from __future__ import annotations

from pathlib import Path

import dspy
from dspy.flex import Flex


class Doubler(dspy.Signature):
    """Return double the input value."""

    value: int = dspy.InputField()
    result: int = dspy.OutputField()


def test_baseline_is_lm_free_and_deterministic() -> None:
    # No LM configured anywhere: construction must not make any LM call.
    a = Flex(Doubler)
    b = Flex(Doubler)

    # Byte-identical baseline source for the same signature.
    assert a.module_src == b.module_src

    # It's the typed dspy.Predict baseline (a dspy.Module subclass) that unwraps the declared output.
    assert "class DoublerModule(dspy.Module)" in a.module_src
    assert "dspy.Predict(" in a.module_src
    assert "value: int -> result: int" in a.module_src
    assert "result.result" in a.module_src


def test_baseline_survives_save_load(tmp_path: Path) -> None:
    program = Flex(Doubler)

    path = tmp_path / "doubler.json"
    program.save(path)

    # Reconstruct with no LM at all — load() must rebind the same baseline source.
    reloaded = Flex(Doubler)
    reloaded.load(path)
    assert reloaded.module_src.strip() == program.module_src.strip()
    assert "dspy.Predict(" in reloaded.module_src
    assert "value: int -> result: int" in reloaded.module_src
