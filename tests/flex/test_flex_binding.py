"""dspy.Flex binds a module's source (``module_src``) and makes it usable like any dspy.Module.

Binding a source — the deterministic baseline at construction, or a GEPA-optimized decomposition —
attaches its predictors flat on the module (so they are discoverable and optimizable), writes nothing
to disk, and round-trips through Module.save / load so the generated code can be persisted and rerun.
Binding runs the source through the interpreter (Flex has no in-process mode), so the tests that
attach or run predictors need Deno; the pure ``module_src`` check uses a MockInterpreter instead.
(The baseline's RLM-vs-Predict shape is covered in test_tools.py; GEPA's code rewriting in
test_flex_gepa.py.)
"""

from __future__ import annotations

import shutil
import textwrap

import pytest

import dspy
from dspy.flex import Flex
from dspy.utils.dummies import DummyLM
from tests.mock_interpreter import MockInterpreter

deno_required = pytest.mark.skipif(shutil.which("deno") is None, reason="Deno is not installed")

# A plain dspy.Predict module we bind and run with a DummyLM, standing in for a GEPA-optimized
# decomposition — end-to-end forward tests bind this instead of the construction baseline.
ECHO_MODULE = textwrap.dedent("""
    class EchoModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.echo = dspy.Predict("q -> a")

        def forward(self, q):
            out = self.echo(q=q)
            return dspy.Prediction(a=out.a)
""").strip()


class Echo(dspy.Signature):
    """Echo the question as the answer."""

    q: str = dspy.InputField()
    a: str = dspy.OutputField()


@deno_required
def test_predictors_are_attached_and_discoverable() -> None:
    with Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter()) as program:
        # The baseline attaches a predictor named `predict` directly onto the module.
        assert hasattr(program, "predict")
        names = [n for n, _ in program.named_predictors()]
        assert "predict" in names


def test_construction_writes_nothing_to_disk(tmp_path, monkeypatch) -> None:
    # The implementation lives in memory; nothing is written until you call save(). Only module_src is
    # inspected here, so a MockInterpreter keeps it Deno-free.
    monkeypatch.chdir(tmp_path)
    program = Flex(Echo, interpreter_factory=lambda: MockInterpreter())
    assert program.module_src is not None
    assert "dspy.Predict(" in program.module_src
    # Flex itself persists nothing — no module file written until save() is called.
    # (dspy's global `.dspy_cache` may appear; that's unrelated to Flex persistence.)
    assert not list(tmp_path.glob("*.py"))
    assert not list(tmp_path.glob("*.json"))


@deno_required
def test_save_load_roundtrips_generated_code(tmp_path) -> None:
    """Module.save/load carries the generated code (module_src), like instruction state."""
    path = tmp_path / "program.json"
    with Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter()) as program:
        program._bind_code(ECHO_MODULE)  # stand in for a GEPA-optimized decomposition
        program.save(path)

    # A fresh Flex starts on the Predict baseline...
    with Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter()) as reloaded:
        assert "self.echo" not in reloaded.module_src
        # ...and load() rebinds the saved code (no LM needed for binding).
        reloaded.load(path)
        assert "self.echo" in reloaded.module_src
        assert hasattr(reloaded, "echo")


@deno_required
def test_end_to_end_forward_after_save_load(tmp_path) -> None:
    """End-to-end: bind a plain dspy.Predict class, save it, reload into a fresh Flex, run it."""
    path = tmp_path / "program.json"
    with Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter()) as program:
        program._bind_code(ECHO_MODULE)
        program.save(path)

    dspy.configure(lm=DummyLM([{"a": "echoed-back"}]))
    with Flex(Echo, interpreter_factory=lambda: dspy.PythonInterpreter()) as reloaded:
        reloaded.load(path)
        assert "self.echo" in reloaded.module_src
        result = reloaded(q="hello")
        assert isinstance(result, dspy.Prediction)
        assert result.a == "echoed-back"
