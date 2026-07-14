from __future__ import annotations

import textwrap

import dspy
from dspy.flex import Flex
from dspy.utils.dummies import DummyLM

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


def test_construction_binds_predict_baseline() -> None:
    # Construction is LM-free: with no tools it binds the deterministic dspy.Predict baseline.
    program = Flex(Echo)
    assert program.module_src is not None
    assert "class EchoModule(dspy.Module)" in program.module_src
    assert "dspy.Predict(" in program.module_src
    assert "q: str -> a: str" in program.module_src
    assert "def forward" in program.module_src
    assert "result.a" in program.module_src  # unwraps the declared output


def test_predictors_are_attached_and_discoverable() -> None:
    program = Flex(Echo)
    # The baseline attaches a predictor named `predict` directly onto the module.
    assert hasattr(program, "predict")
    names = [n for n, _ in program.named_predictors()]
    assert "predict" in names


def test_construction_writes_nothing_to_disk(tmp_path, monkeypatch) -> None:
    # The implementation lives in memory; nothing is written until you call save().
    monkeypatch.chdir(tmp_path)
    program = Flex(Echo)
    assert program.module_src is not None
    assert "dspy.Predict(" in program.module_src
    # Flex itself persists nothing — no module file written until save() is called.
    # (dspy's global `.dspy_cache` may appear; that's unrelated to Flex persistence.)
    assert not list(tmp_path.glob("*.py"))
    assert not list(tmp_path.glob("*.json"))


def test_save_load_roundtrips_generated_code(tmp_path) -> None:
    """Module.save/load carries the generated code (module_src), like instruction state."""
    program = Flex(Echo)
    program._bind_code(ECHO_MODULE)  # stand in for a GEPA-optimized decomposition

    path = tmp_path / "program.json"
    program.save(path)

    # A fresh Flex starts on the Predict baseline...
    reloaded = Flex(Echo)
    assert "self.echo" not in reloaded.module_src
    # ...and load() rebinds the saved code (no LM needed for binding).
    reloaded.load(path)
    assert "self.echo" in reloaded.module_src
    assert hasattr(reloaded, "echo")


def test_end_to_end_forward_after_save_load(tmp_path) -> None:
    """End-to-end: bind a plain dspy.Predict class, save it, reload into a fresh Flex, run it."""
    program = Flex(Echo)
    program._bind_code(ECHO_MODULE)

    path = tmp_path / "program.json"
    program.save(path)

    reloaded = Flex(Echo)
    reloaded.load(path)
    assert "self.echo" in reloaded.module_src

    dspy.configure(lm=DummyLM([{"a": "echoed-back"}]))
    result = reloaded(q="hello")
    assert isinstance(result, dspy.Prediction)
    assert result.a == "echoed-back"
