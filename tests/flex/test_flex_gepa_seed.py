"""GEPA re-optimization seeds from the module's CURRENT source (incl. manual edits).

When a user hand-edits a flexed module (or re-runs GEPA on a previously-optimized one) and
optimizes again, GEPA must build on the current implementation rather than starting over.
``GEPA.compile`` constructs each flex submodule's seed candidate as ``flex.module_src`` (see
gepa.py), reading the live source. These LM-free tests lock in that the current/edited source
is what gets seeded.
"""

from __future__ import annotations

from pathlib import Path

import dspy
from dspy.flex import Flex
from dspy.teleprompt.gepa.gepa_utils import enumerate_flex_submodules, make_code_key

EDITED_MODULE = (
    "class EchoModule(dspy.Module):\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    '        self.echo = dspy.Predict("q -> a")\n'
    "\n"
    "    def forward(self, q):\n"
    "        out = self.echo(q=q)\n"
    "        return dspy.Prediction(a=out.a.upper())"
)


class Echo(dspy.Signature):
    """Echo the question as the answer."""

    q: str = dspy.InputField()
    a: str = dspy.OutputField()


def test_gepa_seed_reflects_hand_edited_persisted_file(tmp_path: Path) -> None:
    path = tmp_path / "echo.py"
    Flex(Echo, persist_to=str(path))  # writes the RLM baseline

    # User hand-edits the persisted module (as they might after a prior GEPA run).
    original = path.read_text(encoding="utf-8")
    edited = original.replace("result.a", "result.a.upper()")
    assert edited != original
    path.write_text(edited, encoding="utf-8")

    # Reconstruct with no LM: the edit loads into module_src.
    reloaded = Flex(Echo, persist_to=str(path))
    assert "result.a.upper()" in reloaded.module_src

    # `flex.module_src` is exactly the value GEPA.compile uses to seed each flex submodule,
    # and exactly what build_program rebinds — so the edit carries straight into the next run.
    assert "result.a.upper()" in reloaded.module_src
    assert "class EchoModule(dspy.Module)" in reloaded.module_src


def test_gepa_discovers_edited_submodule_in_a_program(tmp_path: Path) -> None:
    """Mirror GEPA's discovery: enumerate_flex_submodules + the per-submodule seed value."""
    path = tmp_path / "echo.py"
    flex = Flex(Echo, persist_to=str(path))

    # Apply a manual edit directly (e.g. an in-session tweak before re-optimizing).
    flex._bind_code(EDITED_MODULE)

    class Program(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = flex

        def forward(self, **kwargs):
            return self.classifier(**kwargs)

    program = Program()
    submodules = enumerate_flex_submodules(program)
    assert submodules, "the flex submodule should be discoverable by GEPA"

    seed_candidate: dict[str, str] = {}
    for sub_path, sub in submodules.items():
        seed_candidate[make_code_key(sub_path)] = sub.module_src  # the exact GEPA seed expression

    # Every seeded candidate carries the manual edit forward into the next GEPA run.
    assert seed_candidate
    assert all("out.a.upper()" in code for code in seed_candidate.values())
