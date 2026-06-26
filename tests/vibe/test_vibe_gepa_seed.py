"""GEPA re-optimization seeds from the module's CURRENT source (incl. manual edits).

When a user hand-edits a vibed module (or re-runs GEPA on a previously-optimized one) and
optimizes again, GEPA must build on the current implementation rather than starting over.
``GEPA.compile`` constructs each vibe submodule's seed candidate as
``join_module_code(vibe.predictors_src, vibe.forward_src)`` (see gepa.py), reading the live
source. These LM-free tests lock in that the current/edited source is what gets seeded.
"""

from __future__ import annotations

from pathlib import Path

import dspy
from dspy.teleprompt.gepa.gepa_utils import (
    enumerate_vibe_submodules,
    join_module_code,
    make_code_key,
    split_module_code,
)
from dspy.vibe import Vibe


class Echo(dspy.Signature):
    """Echo the question as the answer."""

    q: str = dspy.InputField()
    a: str = dspy.OutputField()


def test_gepa_seed_reflects_hand_edited_persisted_file(tmp_path: Path) -> None:
    path = tmp_path / "echo.py"
    Vibe(Echo, persist_to=str(path), check_intent=False)  # writes the RLM baseline

    # User hand-edits the persisted module (as they might after a prior GEPA run).
    original = path.read_text(encoding="utf-8")
    edited = original.replace("result.a", "result.a.upper()")
    assert edited != original
    path.write_text(edited, encoding="utf-8")

    # Reconstruct with no LM: the edit loads into predictors_src / forward_src.
    reloaded = Vibe(Echo, persist_to=str(path), check_intent=False)
    assert "result.a.upper()" in reloaded.forward_src

    # This is the EXACT expression GEPA.compile uses to seed each vibe submodule.
    seed_value = join_module_code(reloaded.predictors_src, reloaded.forward_src)
    assert "result.a.upper()" in seed_value

    # And it round-trips back to the (predictors_src, forward_src) build_program rebinds.
    pred, fwd = split_module_code(seed_value)
    assert pred.strip() == reloaded.predictors_src.strip()
    assert fwd.strip() == reloaded.forward_src.strip()


def test_gepa_discovers_edited_submodule_in_a_program(tmp_path: Path) -> None:
    """Mirror GEPA's discovery: enumerate_vibe_submodules + the per-submodule seed value."""
    path = tmp_path / "echo.py"
    vibe = Vibe(Echo, persist_to=str(path), check_intent=False)

    # Apply a manual edit directly (e.g. an in-session tweak before re-optimizing).
    edited_pred = 'PREDICTORS = {\n    "echo": dspy.Predict("q -> a"),\n}'
    edited_fwd = "def forward(self, q):\n    out = self.echo(q=q)\n    return dspy.Prediction(a=out.a.upper())"
    vibe._bind_code(edited_pred, edited_fwd)

    class Program(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = vibe

        def forward(self, **kwargs):
            return self.classifier(**kwargs)

    program = Program()
    submodules = enumerate_vibe_submodules(program)
    assert submodules, "the vibe submodule should be discoverable by GEPA"

    seed_candidate: dict[str, str] = {}
    for sub_path, sub in submodules.items():
        seed_candidate[make_code_key(sub_path)] = join_module_code(sub.predictors_src, sub.forward_src)

    # Every seeded candidate carries the manual edit forward into the next GEPA run.
    assert seed_candidate
    assert all("out.a.upper()" in code for code in seed_candidate.values())
