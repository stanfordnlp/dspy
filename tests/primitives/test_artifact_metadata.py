import copy

import cloudpickle
import orjson
import pytest

import dspy


class ArtifactProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")


class MetadataParameterProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.metadata = dspy.Predict("question -> answer")


def make_program(kind):
    if kind == "module":
        return ArtifactProgram()
    return dspy.Predict("question -> answer")


def artifact_payload():
    return {
        "optimization_run_id": "run-123",
        "context": {
            "attempt": 2,
            "labels": ["candidate", None, True],
            "score": 0.75,
        },
    }


@pytest.mark.parametrize("kind", ["module", "predict"])
def test_artifact_metadata_is_a_mutable_mapping(kind):
    program = make_program(kind)

    assert program.artifact_metadata == {}

    metadata = program.artifact_metadata
    metadata["cmpnd"] = artifact_payload()

    assert program.artifact_metadata == {"cmpnd": artifact_payload()}


def test_artifact_metadata_assignment_validates_copies_and_replaces():
    program = ArtifactProgram()
    assigned = {"cmpnd": artifact_payload()}

    program.artifact_metadata = assigned
    assigned["cmpnd"]["context"]["labels"].append("assignment mutation")

    assert program.artifact_metadata == {"cmpnd": artifact_payload()}

    with pytest.raises(TypeError):
        program.artifact_metadata = {"cmpnd": "run-123"}

    assert program.artifact_metadata == {"cmpnd": artifact_payload()}


@pytest.mark.parametrize("kind", ["module", "predict"])
def test_artifact_metadata_round_trips_through_raw_state(kind):
    source = make_program(kind)
    source.artifact_metadata["cmpnd"] = artifact_payload()
    source.artifact_metadata["registry"] = {"model_version": "v2"}

    state = source.dump_state()

    assert state["metadata"]["artifact_metadata"] == {
        "cmpnd": artifact_payload(),
        "registry": {"model_version": "v2"},
    }

    target = make_program(kind)
    target.artifact_metadata["existing"] = {"value": "old"}
    target.load_state(state)

    assert target.artifact_metadata == {
        "cmpnd": artifact_payload(),
        "registry": {"model_version": "v2"},
    }
    if kind == "predict":
        assert not hasattr(target, "metadata")


@pytest.mark.parametrize("kind", ["module", "predict"])
def test_empty_artifact_metadata_is_omitted_and_missing_metadata_clears(kind):
    source = make_program(kind)

    state = source.dump_state()

    assert "metadata" not in state

    target = make_program(kind)
    target.artifact_metadata["cmpnd"] = artifact_payload()
    target.load_state(state)

    assert target.artifact_metadata == {}


@pytest.mark.parametrize("kind", ["module", "predict"])
def test_dumped_and_loaded_artifact_metadata_are_detached(kind):
    source = make_program(kind)
    source.artifact_metadata["cmpnd"] = artifact_payload()

    state = source.dump_state()
    state["metadata"]["artifact_metadata"]["cmpnd"]["context"]["labels"].append("dump mutation")

    assert source.artifact_metadata == {"cmpnd": artifact_payload()}

    load_state = source.dump_state()
    target = make_program(kind)
    target.load_state(load_state)
    load_state["metadata"]["artifact_metadata"]["cmpnd"]["context"]["labels"].append("load mutation")

    assert target.artifact_metadata == {"cmpnd": artifact_payload()}


@pytest.mark.parametrize("kind", ["module", "predict"])
@pytest.mark.parametrize("suffix", [".json", ".pkl"])
def test_artifact_metadata_round_trips_through_state_only_files(kind, suffix, tmp_path):
    source = make_program(kind)
    source.artifact_metadata["cmpnd"] = artifact_payload()
    path = tmp_path / f"program{suffix}"

    source.save(path)

    if suffix == ".json":
        saved_state = orjson.loads(path.read_bytes())
    else:
        with path.open("rb") as file:
            saved_state = cloudpickle.load(file)
    assert saved_state["metadata"]["artifact_metadata"] == {"cmpnd": artifact_payload()}
    assert "dependency_versions" in saved_state["metadata"]

    target = make_program(kind)
    target.artifact_metadata["existing"] = {"value": "old"}
    target.load(path, allow_pickle=suffix == ".pkl")

    assert target.artifact_metadata == {"cmpnd": artifact_payload()}
    if kind == "predict":
        assert not hasattr(target, "metadata")


@pytest.mark.parametrize("suffix", [".json", ".pkl"])
def test_empty_artifact_metadata_is_omitted_from_state_only_files(suffix, tmp_path):
    program = ArtifactProgram()
    path = tmp_path / f"program{suffix}"

    program.save(path)

    if suffix == ".json":
        saved_state = orjson.loads(path.read_bytes())
    else:
        with path.open("rb") as file:
            saved_state = cloudpickle.load(file)
    assert "artifact_metadata" not in saved_state["metadata"]


@pytest.mark.parametrize("kind", ["module", "predict"])
def test_artifact_metadata_round_trips_through_full_program_save(kind, tmp_path):
    source = make_program(kind)
    source.artifact_metadata["cmpnd"] = artifact_payload()
    path = tmp_path / "program"

    source.save(path, save_program=True)
    loaded = dspy.load(path, allow_pickle=True)

    assert loaded.artifact_metadata == {"cmpnd": artifact_payload()}


def test_full_program_save_validates_artifact_metadata(tmp_path):
    program = ArtifactProgram()
    program.predict.artifact_metadata["cmpnd"] = {"value": object()}

    with pytest.raises(TypeError):
        program.save(tmp_path / "program", save_program=True)


def test_metadata_is_a_reserved_root_state_key():
    program = MetadataParameterProgram()

    with pytest.raises(ValueError, match="reserved for DSPy serialization metadata"):
        program.dump_state()


@pytest.mark.parametrize(
    ("namespace", "payload", "exception"),
    [
        (1, {}, TypeError),
        ("", {}, ValueError),
        ("cmpnd", "run-123", TypeError),
        ("cmpnd", {1: "value"}, TypeError),
        ("cmpnd", {"value": object()}, TypeError),
        ("cmpnd", {"score": float("nan")}, ValueError),
        ("cmpnd", {"score": float("inf")}, ValueError),
    ],
)
def test_dump_state_validates_artifact_metadata_as_namespaced_json_objects(namespace, payload, exception):
    program = ArtifactProgram()
    program.artifact_metadata[namespace] = payload

    with pytest.raises(exception):
        program.dump_state()


@pytest.mark.parametrize("kind", ["module", "predict"])
@pytest.mark.parametrize(
    ("artifact_metadata", "exception"),
    [
        (None, TypeError),
        ({1: {}}, TypeError),
        ({"": {}}, ValueError),
        ({"cmpnd": "run-123"}, TypeError),
        ({"cmpnd": {"value": object()}}, TypeError),
        ({"cmpnd": {"score": float("-inf")}}, ValueError),
    ],
)
def test_load_state_rejects_invalid_artifact_metadata_without_mutation(kind, artifact_metadata, exception):
    source = make_program(kind)
    state = source.dump_state()
    state["metadata"] = {"artifact_metadata": artifact_metadata}

    target = make_program(kind)
    target.artifact_metadata["existing"] = {"value": "old"}

    with pytest.raises(exception):
        target.load_state(state)

    assert target.artifact_metadata == {"existing": {"value": "old"}}


@pytest.mark.parametrize("kind", ["module", "predict"])
def test_failed_parameter_load_preserves_artifact_metadata(kind):
    source = make_program(kind)
    source.artifact_metadata["cmpnd"] = artifact_payload()
    state = source.dump_state()
    if kind == "module":
        del state["predict"]
    else:
        del state["signature"]

    target = make_program(kind)
    target.artifact_metadata["existing"] = {"value": "old"}
    original_metadata = copy.deepcopy(target.artifact_metadata)

    with pytest.raises(KeyError):
        target.load_state(state)

    assert target.artifact_metadata == original_metadata
