from __future__ import annotations

import json
import textwrap

import dspy
from dspy.flex import flex
from dspy.flex.exploration import ExplorationStore, candidate_id
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

    @flex(persist_to=str(tmp_path / "echo_flex.py"), intent_check="off")
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

    @flex(persist_to=str(tmp_path / "echo_flex.py"), intent_check="off")
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

    @flex(persist_to=str(persist_path), intent_check="off")
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

    @flex(persist_to=str(persist_path), intent_check="off")
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


def test_manual_edit_is_honored_when_signature_unchanged(tmp_path) -> None:
    persist_path = tmp_path / "echo_flex.py"
    dspy.configure(lm=_make_codegen_lm())

    @flex(persist_to=str(persist_path), intent_check="off")
    class Echo(dspy.Signature):
        """Echo."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    Echo()

    # Hand-edit the forward body. Signature (and its hash) is untouched, so only
    # the body hash goes stale.
    original = persist_path.read_text()
    edited = original.replace("dspy.Prediction(a=out.a)", "dspy.Prediction(a=out.a.upper())")
    assert edited != original
    persist_path.write_text(edited)

    # Reconstruct with NO codegen LM available — the edit must be honored, not
    # regenerated.
    dspy.configure(lm=DummyLM([]))
    program = Echo()
    assert "out.a.upper()" in program.forward_src

    # A `manual_edit` event was recorded, and the file's body hash was refreshed
    # so the file reads as pristine again.
    history = ExplorationStore(tmp_path, "Echo").get_history()
    assert any(e["event"] == "manual_edit" for e in history)

    refreshed = persist_path.read_text()
    expected_body_hash = candidate_id(program.predictors_src, program.forward_src)
    assert f"# __FLEX_BODY_HASH__: {expected_body_hash}" in refreshed

    # The edit is the deployed artifact, so it's appended as a manifest version
    # (after the initial codegen) and marked as a manual edit.
    manifest = json.loads((tmp_path / ".flex" / "manifest.json").read_text())
    versions = manifest["flex_modules"]["Echo"]["versions"]
    assert len(versions) == 2
    assert versions[-1]["notes"] == "manual edit"
    assert versions[-1]["candidate_id"] == expected_body_hash


def test_signature_change_seeds_regeneration_from_current_implementation(tmp_path) -> None:
    persist_path = tmp_path / "echo_flex.py"
    dspy.configure(lm=_make_codegen_lm())

    @flex(persist_to=str(persist_path), intent_check="off")
    class Echo(dspy.Signature):
        """Echo."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    Echo()
    seed_id = candidate_id(CANNED_PREDICTORS, CANNED_FORWARD)

    # Same flex_id (class name "Echo") + same persist_to, but a changed signature
    # → hash mismatch → regenerate, seeded from the current on-disk body.
    new_predictors = CANNED_PREDICTORS.replace('"echo"', '"echo2"')
    new_forward = CANNED_FORWARD.replace("self.echo", "self.echo2")
    dspy.configure(lm=DummyLM([{"predictors_src": new_predictors, "forward_src": new_forward}]))

    @flex(persist_to=str(persist_path), intent_check="off")
    class Echo(dspy.Signature):  # noqa: F811
        """Echo."""

        q: str = dspy.InputField()
        extra: str = dspy.InputField()
        a: str = dspy.OutputField()

    program = Echo()
    assert "echo2" in program.predictors_src

    # The regeneration's codegen event is parented to the prior implementation.
    history = ExplorationStore(tmp_path, "Echo").get_history()
    codegen_events = [e for e in history if e["event"] == "codegen"]
    assert any(e.get("parents") == [seed_id] for e in codegen_events)


def test_legacy_file_without_body_hash_loads_and_backfills(tmp_path) -> None:
    persist_path = tmp_path / "echo_flex.py"
    dspy.configure(lm=_make_codegen_lm())

    @flex(persist_to=str(persist_path), intent_check="off")
    class Echo(dspy.Signature):
        """Echo."""

        q: str = dspy.InputField()
        a: str = dspy.OutputField()

    Echo()

    # Simulate a file written before body hashes existed by stripping the line.
    text = persist_path.read_text()
    legacy = "\n".join(
        line for line in text.splitlines() if not line.startswith("# __FLEX_BODY_HASH__:")
    )
    persist_path.write_text(legacy)
    assert "# __FLEX_BODY_HASH__:" not in persist_path.read_text()

    dspy.configure(lm=DummyLM([]))
    program = Echo()
    assert "PREDICTORS" in program.predictors_src
    # The hash is backfilled on load so future edits become detectable.
    assert "# __FLEX_BODY_HASH__:" in persist_path.read_text()


def test_in_memory_only_mode_binds_without_writing_disk(tmp_path) -> None:
    dspy.configure(lm=_make_codegen_lm())

    @flex(intent_check="off")
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

    @flex(persist_to=str(tmp_path / "echo_flex.py"), intent_check="off")
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

    @flex(persist_to=str(persist_path), intent_check="off")
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
