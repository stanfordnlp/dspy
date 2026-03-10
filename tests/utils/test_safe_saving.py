import sys
from pathlib import Path
from unittest.mock import patch

import orjson
import pytest

import dspy
from dspy.signatures.signature import _str_to_type, _type_to_str
from dspy.utils.saving import get_dependency_versions

# ---------------------------------------------------------------------------
# Type serialization round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("type_", "expected_str"),
    [
        (str, "str"),
        (int, "int"),
        (float, "float"),
        (bool, "bool"),
        (list, "list"),
        (dict, "dict"),
        (type(None), "NoneType"),
        (list[str], "list[str]"),
        (dict[str, int], "dict[str, int]"),
    ],
)
def test_type_round_trip(type_, expected_str):
    serialized = _type_to_str(type_)
    assert serialized == expected_str
    assert _str_to_type(serialized) == type_


# ---------------------------------------------------------------------------
# Signature enhanced dump_state / load_state
# ---------------------------------------------------------------------------


def test_signature_dump_state_includes_field_metadata():
    sig = dspy.Signature("question: str, context: list[str] -> answer: int")
    state = sig.dump_state()

    assert len(state["fields"]) == 3

    q = state["fields"][0]
    assert q["name"] == "question"
    assert q["field_type"] == "input"
    assert q["type"] == "str"

    ctx = state["fields"][1]
    assert ctx["name"] == "context"
    assert ctx["field_type"] == "input"
    assert ctx["type"] == "list[str]"

    ans = state["fields"][2]
    assert ans["name"] == "answer"
    assert ans["field_type"] == "output"
    assert ans["type"] == "int"


def test_signature_load_state_reconstructs_from_scratch():
    """When field names don't match, load_state rebuilds from the saved metadata."""
    original = dspy.Signature("question, context -> answer")
    state = original.dump_state()

    # Use a totally different base signature.
    placeholder = dspy.Signature("x -> y")
    loaded = placeholder.load_state(state)

    assert list(loaded.fields.keys()) == ["question", "context", "answer"]
    assert list(loaded.input_fields.keys()) == ["question", "context"]
    assert list(loaded.output_fields.keys()) == ["answer"]
    assert loaded.instructions == original.instructions


def test_signature_load_state_updates_in_place_when_fields_match():
    """When field names match, load_state updates prefix/desc in place."""
    sig = dspy.Signature("q -> a")
    state = sig.dump_state()

    # Modify the prefix in the saved state.
    state["fields"][0]["prefix"] = "Custom Question:"

    loaded = sig.load_state(state)
    assert loaded.fields["q"].json_schema_extra["prefix"] == "Custom Question:"


def test_signature_load_state_backward_compat_legacy_format():
    """Legacy state without 'name' key still loads correctly."""
    sig = dspy.Signature("q -> a")
    legacy_state = {
        "instructions": "Legacy instructions",
        "fields": [
            {"prefix": "Q:", "description": "the question"},
            {"prefix": "A:", "description": "the answer"},
        ],
    }

    loaded = sig.load_state(legacy_state)
    assert loaded.instructions == "Legacy instructions"
    assert loaded.fields["q"].json_schema_extra["prefix"] == "Q:"
    assert loaded.fields["a"].json_schema_extra["desc"] == "the answer"


# ---------------------------------------------------------------------------
# Predict config in state
# ---------------------------------------------------------------------------


def test_predict_dump_state_includes_config():
    predict = dspy.Predict("q -> a", temperature=0.7, max_tokens=100)
    state = predict.dump_state()

    assert "config" in state
    assert state["config"] == {"temperature": 0.7, "max_tokens": 100}


def test_predict_load_state_restores_config():
    predict = dspy.Predict("q -> a", temperature=0.7)
    state = predict.dump_state()

    new_predict = dspy.Predict("q -> a")
    assert new_predict.config == {}
    new_predict.load_state(state)
    assert new_predict.config == {"temperature": 0.7}


def test_predict_load_state_backward_compat_no_config():
    """State from older versions without 'config' keeps the existing config."""
    predict = dspy.Predict("q -> a")
    state = predict.dump_state()
    del state["config"]

    new_predict = dspy.Predict("q -> a", my_param=42)
    new_predict.load_state(state)
    assert new_predict.config == {"my_param": 42}


# ---------------------------------------------------------------------------
# Safe program save / load
# ---------------------------------------------------------------------------


def test_safe_save_creates_correct_files(tmp_path):
    cot = dspy.ChainOfThought("question -> answer")
    save_dir = tmp_path / "model"

    cot.save(str(save_dir), save_program=True, safe=True)

    assert (save_dir / "metadata.json").exists()
    assert (save_dir / "program.json").exists()
    assert not (save_dir / "program.pkl").exists()

    metadata = orjson.loads((save_dir / "metadata.json").read_bytes())
    assert metadata["format"] == "safe_v1"
    assert "dependency_versions" in metadata


def test_safe_save_program_json_structure(tmp_path):
    cot = dspy.ChainOfThought("question -> answer")
    save_dir = tmp_path / "model"

    cot.save(str(save_dir), save_program=True, safe=True)

    program = orjson.loads((save_dir / "program.json").read_bytes())

    assert program["module_class"] == "dspy.predict.chain_of_thought.ChainOfThought"
    assert len(program["module_tree"]) == 1
    assert program["module_tree"][0]["path"] == "predict"
    assert program["module_tree"][0]["class"] == "dspy.predict.predict.Predict"
    assert "predict" in program["state"]


def test_safe_round_trip_chain_of_thought(tmp_path):
    cot = dspy.ChainOfThought("question -> answer")
    cot.predict.signature = cot.predict.signature.with_instructions("You are a helpful assistant.")
    cot.predict.demos = [
        dspy.Example(question="What is 2+2?", answer="4", reasoning="Basic math").with_inputs("question"),
    ]

    save_dir = tmp_path / "model"
    cot.save(str(save_dir), save_program=True, safe=True)

    loaded = dspy.load(str(save_dir))

    assert isinstance(loaded, dspy.ChainOfThought)
    assert loaded.predict.signature.instructions == "You are a helpful assistant."
    assert loaded.predict.signature.signature == cot.predict.signature.signature
    assert list(loaded.predict.signature.fields.keys()) == list(cot.predict.signature.fields.keys())
    assert len(loaded.predict.demos) == 1
    assert loaded.predict.demos[0]["question"] == "What is 2+2?"


def test_safe_round_trip_predict(tmp_path):
    predict = dspy.Predict("question -> answer", temperature=0.5)
    predict.demos = [
        dspy.Example(question="Hi", answer="Hello").with_inputs("question"),
    ]

    save_dir = tmp_path / "model"
    predict.save(str(save_dir), save_program=True, safe=True)

    loaded = dspy.load(str(save_dir))

    assert isinstance(loaded, dspy.Predict)
    assert list(loaded.signature.fields.keys()) == ["question", "answer"]
    assert loaded.config == {"temperature": 0.5}
    assert len(loaded.demos) == 1


def test_safe_round_trip_nested_custom_module(tmp_path):
    # Create a custom module file so it can be imported by class path.
    custom_module_path = tmp_path / "custom_module.py"
    custom_module_path.write_text(
        """\
import dspy

class InnerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.predict(question=question)

class OuterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.inner = InnerModule()
        self.summarize = dspy.Predict("text -> summary")

    def forward(self, question, text):
        answer = self.inner(question=question)
        summary = self.summarize(text=text)
        return answer, summary
"""
    )

    sys.path.insert(0, str(tmp_path))
    try:
        import custom_module

        module = custom_module.OuterModule()
        module.inner.predict.demos = [
            dspy.Example(question="Hi", answer="Hello").with_inputs("question"),
        ]
        module.summarize.demos = [
            dspy.Example(text="Long text", summary="Short").with_inputs("text"),
        ]

        save_dir = tmp_path / "nested_model"
        module.save(str(save_dir), save_program=True, safe=True)

        loaded = dspy.load(str(save_dir))

        assert isinstance(loaded, custom_module.OuterModule)
        assert isinstance(loaded.inner, custom_module.InnerModule)
        assert isinstance(loaded.inner.predict, dspy.Predict)
        assert isinstance(loaded.summarize, dspy.Predict)
        assert list(loaded.inner.predict.signature.fields.keys()) == ["question", "answer"]
        assert list(loaded.summarize.signature.fields.keys()) == ["text", "summary"]
        assert len(loaded.inner.predict.demos) == 1
        assert loaded.inner.predict.demos[0]["question"] == "Hi"
        assert len(loaded.summarize.demos) == 1
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("custom_module", None)


def test_safe_load_does_not_require_allow_pickle(tmp_path):
    """Safe format loads without allow_pickle=True."""
    predict = dspy.Predict("q -> a")
    save_dir = tmp_path / "model"
    predict.save(str(save_dir), save_program=True, safe=True)

    loaded = dspy.load(str(save_dir))
    assert isinstance(loaded, dspy.Predict)


def test_legacy_load_still_requires_allow_pickle(tmp_path):
    """Legacy cloudpickle format still requires allow_pickle=True."""
    predict = dspy.Predict("q -> a")
    save_dir = tmp_path / "legacy_model"
    predict.save(str(save_dir), save_program=True)

    with pytest.raises(ValueError, match="allow_pickle"):
        dspy.load(str(save_dir))

    loaded = dspy.load(str(save_dir), allow_pickle=True)
    assert isinstance(loaded, dspy.Predict)


def test_safe_load_version_mismatch_warning(tmp_path):
    import logging

    from dspy.utils.saving import logger as saving_logger

    predict = dspy.Predict("q -> a")
    save_dir = tmp_path / "model"
    predict.save(str(save_dir), save_program=True, safe=True)

    class ListHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    handler = ListHandler()
    original_level = saving_logger.level
    saving_logger.addHandler(handler)
    saving_logger.setLevel(logging.WARNING)

    try:
        mismatch_versions = {"python": "3.9", "dspy": "2.4.0", "cloudpickle": "2.0"}
        with patch("dspy.utils.saving.get_dependency_versions", return_value=mismatch_versions):
            loaded = dspy.load(str(save_dir))

        assert isinstance(loaded, dspy.Predict)
        # Should have logged warnings for version mismatches.
        assert len(handler.messages) >= 1
        assert any("mismatch" in msg for msg in handler.messages)
    finally:
        saving_logger.setLevel(original_level)
        saving_logger.removeHandler(handler)


def test_safe_save_rejects_non_directory_path(tmp_path):
    predict = dspy.Predict("q -> a")

    with pytest.raises(ValueError, match="directory"):
        predict.save(str(tmp_path / "model.json"), save_program=True, safe=True)


def test_module_tree_collection():
    cot = dspy.ChainOfThought("q -> a")
    tree = cot._collect_module_tree()

    assert len(tree) == 1
    assert tree[0]["path"] == "predict"
    assert tree[0]["class"] == "dspy.predict.predict.Predict"


def test_safe_load_missing_class_raises():
    """If the saved class is not importable, safe load raises ImportError."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "model"
        save_dir.mkdir()

        metadata = {
            "format": "safe_v1",
            "dependency_versions": get_dependency_versions(),
        }
        (save_dir / "metadata.json").write_bytes(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))

        program = {
            "module_class": "nonexistent.module.FakeClass",
            "module_tree": [],
            "state": {},
        }
        (save_dir / "program.json").write_bytes(orjson.dumps(program, option=orjson.OPT_INDENT_2))

        with pytest.raises(ModuleNotFoundError):
            dspy.load(str(save_dir))
