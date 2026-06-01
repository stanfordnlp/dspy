from __future__ import annotations

import json
import textwrap

import dspy
from dspy.flex import flex
from dspy.utils.dummies import DummyLM

CANNED_PREDICTORS = textwrap.dedent("""
    PREDICTORS = {
        "echo": dspy.Predict("q -> a"),
    }
""").strip()

CANNED_FORWARD = textwrap.dedent("""
    def forward(self, q):
        out = self.echo(q=q)
        return dspy.Prediction(a=out.a)
""").strip()


def _make_codegen_lm():
    return DummyLM([{"predictors_src": CANNED_PREDICTORS, "forward_src": CANNED_FORWARD}])


def test_decorator_construction_runs_codegen_and_binds_forward(tmp_path) -> None:
    dspy.configure(lm=_make_codegen_lm())

    @flex(persist_to=str(tmp_path / "echo_flex.py"))
    class Echo(dspy.Signature):
        """Echo the question as the answer, twice."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    program = Echo()
    assert program.predictors_src is not None
    assert program.forward_src is not None
    assert "PREDICTORS" in program.predictors_src
    assert "def forward" in program.forward_src


def test_predictors_are_attached_and_discoverable(tmp_path) -> None:
    dspy.configure(lm=_make_codegen_lm())

    @flex(persist_to=str(tmp_path / "echo_flex.py"))
    class Echo(dspy.Signature):
        """Echo."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    program = Echo()
    names = [n for n, _ in program.named_predictors()]
    assert "echo" in names


def test_persisted_file_is_written_and_reloaded(tmp_path) -> None:
    persist_path = tmp_path / "echo_flex.py"
    dspy.configure(lm=_make_codegen_lm())

    @flex(persist_to=str(persist_path))
    class Echo(dspy.Signature):
        """Echo."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    Echo()
    assert persist_path.exists()
    text = persist_path.read_text()
    assert "__FLEX_SIGNATURE_HASH__" in text
    assert "PREDICTORS" in text
    assert "def forward" in text

    # Re-construct with NO LM configured — must load from disk without LM call.
    dspy.configure(lm=DummyLM([]))
    program2 = Echo()
    assert program2.predictors_src is not None
    assert "PREDICTORS" in program2.predictors_src


def test_signature_hash_mismatch_triggers_regeneration(tmp_path) -> None:
    persist_path = tmp_path / "echo_flex.py"
    dspy.configure(lm=_make_codegen_lm())

    @flex(persist_to=str(persist_path))
    class Echo(dspy.Signature):
        """Echo."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    Echo()
    original = persist_path.read_text()
    tampered = original.replace("__FLEX_SIGNATURE_HASH__: ", "__FLEX_SIGNATURE_HASH__: nope_")
    persist_path.write_text(tampered)

    new_predictors = CANNED_PREDICTORS.replace('"echo"', '"echo2"')
    new_forward = CANNED_FORWARD.replace("self.echo", "self.echo2")
    dspy.configure(lm=DummyLM([{"predictors_src": new_predictors, "forward_src": new_forward}]))

    program = Echo()
    assert "echo2" in program.predictors_src


def test_in_memory_only_mode_binds_without_writing_disk(tmp_path) -> None:
    dspy.configure(lm=_make_codegen_lm())

    @flex
    class Echo(dspy.Signature):
        """Echo."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    program = Echo()
    assert program.predictors_src is not None
    assert program.forward_src is not None
    # In-memory mode never writes the manifest (no persist_to). It DOES still
    # record codegen events in the exploration log under .flex/Echo/.
    assert not (tmp_path / ".flex" / "manifest.json").exists()


def test_end_to_end_forward_call(tmp_path) -> None:
    lm = DummyLM(
        [
            {"predictors_src": CANNED_PREDICTORS, "forward_src": CANNED_FORWARD},
            {"a": "echoed-back"},
        ]
    )
    dspy.configure(lm=lm)

    @flex(persist_to=str(tmp_path / "echo_flex.py"))
    class Echo(dspy.Signature):
        """Echo."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    program = Echo()
    result = program(q="hello")
    assert isinstance(result, dspy.Prediction)
    assert result.a == "echoed-back"


def test_manifest_records_a_version(tmp_path) -> None:
    persist_path = tmp_path / "echo_flex.py"
    dspy.configure(lm=_make_codegen_lm())

    @flex(persist_to=str(persist_path))
    class Echo(dspy.Signature):
        """Echo."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    Echo()

    manifest_path = tmp_path / ".flex" / "manifest.json"
    assert manifest_path.exists()

    data = json.loads(manifest_path.read_text())
    assert "Echo" in data["flex_modules"]
    versions = data["flex_modules"]["Echo"]["versions"]
    assert len(versions) == 1
    assert versions[0]["id"] == 0
    assert versions[0]["signature_hash"]
