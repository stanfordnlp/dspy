from __future__ import annotations

import textwrap

import dspy
from dspy.utils.dummies import DummyLM
from dspy.vibe import Flex

# A plain dspy.Predict body we can persist and actually run with a DummyLM — the RLM
# baseline that Flex binds at construction is too heavy to execute in a unit test, so
# end-to-end forward tests use this instead.
PREDICT_PREDICTORS = textwrap.dedent("""
    PREDICTORS = {
        "echo": dspy.Predict("q -> a"),
    }
""").strip()

PREDICT_FORWARD = textwrap.dedent("""
    def forward(self, q):
        out = self.echo(q=q)
        return dspy.Prediction(a=out.a)
""").strip()


class Echo(dspy.Signature):
    """Echo the question as the answer."""

    q: str = dspy.InputField()
    a: str = dspy.OutputField()


def test_construction_binds_rlm_baseline(tmp_path) -> None:
    # Construction is LM-free: it binds the deterministic dspy.RLM baseline.
    program = Flex(Echo, persist_to=str(tmp_path / "echo_flex.py"))
    assert program.predictors_src is not None
    assert program.forward_src is not None
    assert "dspy.RLM(" in program.predictors_src
    assert "q: str -> a: str" in program.predictors_src
    assert "def forward" in program.forward_src
    assert "result.a" in program.forward_src  # unwraps the declared output


def test_predictors_are_attached_and_discoverable(tmp_path) -> None:
    program = Flex(Echo, persist_to=str(tmp_path / "echo_flex.py"))
    # The baseline binds an attribute named `rlm`.
    assert hasattr(program, "rlm")
    names = [n for n, _ in program.named_predictors()]
    # named_predictors() surfaces the RLM's internal predictors.
    assert "rlm.generate_action" in names
    assert "rlm.extract" in names


def test_persisted_file_is_written_and_reloaded(tmp_path) -> None:
    persist_path = tmp_path / "echo_flex.py"
    Flex(Echo, persist_to=str(persist_path))
    assert persist_path.exists()
    text = persist_path.read_text()
    assert "__FLEX_SIGNATURE_HASH__" in text
    assert "PREDICTORS" in text
    assert "dspy.RLM(" in text
    assert "def forward" in text
    # Bookkeeping is gone: no body hash, no flex_id, no .flex dir.
    assert "__FLEX_BODY_HASH__" not in text
    assert "flex_id" not in text
    assert not (tmp_path / ".flex").exists()

    # Re-construct with NO LM configured — must load from disk without an LM call.
    dspy.configure(lm=DummyLM([]))
    program2 = Flex(Echo, persist_to=str(persist_path))
    assert program2.predictors_src is not None
    assert "dspy.RLM(" in program2.predictors_src


def test_signature_change_resets_to_fresh_baseline(tmp_path) -> None:
    """A changed signature discards the old body and rebinds a fresh RLM baseline."""
    persist_path = tmp_path / "echo_flex.py"
    Flex(Echo, persist_to=str(persist_path))
    original = persist_path.read_text()
    tampered = original.replace("__FLEX_SIGNATURE_HASH__: ", "__FLEX_SIGNATURE_HASH__: nope_")
    persist_path.write_text(tampered)

    # No LM needed — the reset to baseline is deterministic and LM-free.
    dspy.configure(lm=DummyLM([]))
    program = Flex(Echo, persist_to=str(persist_path))
    assert "dspy.RLM(" in program.predictors_src
    assert "q: str -> a: str" in program.predictors_src


def test_hand_edit_is_loaded_when_signature_unchanged(tmp_path) -> None:
    """A hand-edited body is loaded as-is on the next run (signature unchanged)."""
    persist_path = tmp_path / "echo_flex.py"
    Flex(Echo, persist_to=str(persist_path))

    original = persist_path.read_text()
    edited = original.replace("result.a", "result.a.upper()")
    assert edited != original
    persist_path.write_text(edited)

    # Reconstruct with NO codegen LM available — the edit is loaded, not regenerated.
    dspy.configure(lm=DummyLM([]))
    program = Flex(Echo, persist_to=str(persist_path))
    assert "result.a.upper()" in program.forward_src


def test_in_memory_only_mode_binds_without_writing_disk(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    program = Flex(Echo)  # persist_to=None
    assert program.predictors_src is not None
    assert program.forward_src is not None
    assert "dspy.RLM(" in program.predictors_src
    # No persistence and no bookkeeping directory.
    assert not (tmp_path / ".flex").exists()
    assert not list(tmp_path.glob("*.py"))


def test_end_to_end_forward_call(tmp_path) -> None:
    """End-to-end forward: persist a plain dspy.Predict body and run it.

    The RLM baseline needs a code interpreter, so we write a Predict-based body to the
    persisted file (signature hash intact, so it's loaded as-is) and run it with a DummyLM.
    """
    from dspy.vibe.persistence import parse_persisted_file, render_persisted_file

    persist_path = tmp_path / "echo_flex.py"
    Flex(Echo, persist_to=str(persist_path))

    parsed = parse_persisted_file(persist_path.read_text())
    assert parsed is not None
    persist_path.write_text(
        render_persisted_file(
            signature_hash=parsed.signature_hash,
            signature_name="Echo",
            predictors_src=PREDICT_PREDICTORS,
            forward_src=PREDICT_FORWARD,
        )
    )

    dspy.configure(lm=DummyLM([{"a": "echoed-back"}]))
    program = Flex(Echo, persist_to=str(persist_path))
    assert "self.echo" in program.forward_src
    result = program(q="hello")
    assert isinstance(result, dspy.Prediction)
    assert result.a == "echoed-back"
