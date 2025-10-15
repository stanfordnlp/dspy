"""Tests for API key exclusion in saved programs."""

import json
import tempfile
from pathlib import Path

import dspy


def test_api_key_not_saved_in_json():
    """Test that API key is not saved in JSON format."""
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        model_type="chat",
        temperature=1.0,
        max_tokens=100,
        api_key="sk-test-api-key-12345",
    )

    predict = dspy.Predict("question -> answer")
    predict.lm = lm

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "program.json"
        predict.save(json_path)

        with open(json_path) as f:
            saved_state = json.load(f)

        # Verify API key is not in the saved state
        assert "api_key" not in saved_state.get("lm", {}), "API key should not be saved in JSON"

        # Verify other attributes are saved
        assert saved_state["lm"]["model"] == "openai/gpt-4o-mini"
        assert saved_state["lm"]["temperature"] == 1.0
        assert saved_state["lm"]["max_tokens"] == 100


def test_api_base_not_saved():
    """Test that api_base and base_url are not saved."""
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key="sk-test-key",
        api_base="https://custom-endpoint.com/v1",
        base_url="https://another-endpoint.com",
    )

    predict = dspy.Predict("question -> answer")
    predict.lm = lm

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "program.json"
        predict.save(json_path)

        with open(json_path) as f:
            saved_state = json.load(f)
            lm_state = saved_state.get("lm", {})

        # Verify sensitive keys are not saved
        assert "api_key" not in lm_state
        assert "api_base" not in lm_state
        assert "base_url" not in lm_state


def test_load_without_api_key():
    """Test that programs can be loaded without API keys."""
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        temperature=0.5,
        max_tokens=50,
        api_key="sk-test-key",
    )

    predict = dspy.Predict("question -> answer")
    predict.lm = lm

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "program.json"
        predict.save(json_path)

        # Load the program
        new_predict = dspy.Predict("question -> answer")
        new_predict.load(json_path)

        # Verify LM was loaded
        assert new_predict.lm is not None
        assert new_predict.lm.model == "openai/gpt-4o-mini"
        assert new_predict.lm.kwargs["temperature"] == 0.5
        assert new_predict.lm.kwargs["max_tokens"] == 50

        # Verify API key is not in loaded state
        assert "api_key" not in new_predict.lm.kwargs


def test_api_key_not_saved_in_pickle():
    """Test that API key is not saved in pickle format."""
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key="sk-test-pickle-key",
    )

    predict = dspy.Predict("question -> answer")
    predict.lm = lm

    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = Path(tmpdir) / "program.pkl"
        predict.save(pkl_path)

        # Load and verify
        new_predict = dspy.Predict("question -> answer")
        new_predict.load(pkl_path)

        assert new_predict.lm is not None
        assert new_predict.lm.model == "openai/gpt-4o-mini"
        assert "api_key" not in new_predict.lm.kwargs


def test_save_program_with_api_key():
    """Test saving full program (not just state) with API key."""

    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict("question -> answer")

        def forward(self, question):
            return self.predict(question=question)

    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key="sk-test-full-program-key",
    )

    program = MyProgram()
    program.predict.lm = lm

    with tempfile.TemporaryDirectory() as tmpdir:
        save_dir = Path(tmpdir) / "saved_program"
        program.save(save_dir, save_program=True)

        # Verify metadata.json doesn't contain API key
        metadata_path = save_dir / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Metadata should only have dependency versions, not program state
        assert "dependency_versions" in metadata
        assert "api_key" not in json.dumps(metadata)


def test_existing_functionality_preserved():
    """Test that existing save/load functionality still works."""
    # This is similar to test_save_and_load_with_json in test_base_module.py
    model = dspy.ChainOfThought(dspy.Signature("q -> a"))
    model.predict.signature = model.predict.signature.with_instructions("You are a helpful assistant.")
    model.predict.demos = [
        dspy.Example(q="What is the capital of France?", a="Paris", reasoning="n/a").with_inputs("q"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "model.json"
        model.save(save_path)

        new_model = dspy.ChainOfThought(dspy.Signature("q -> a"))
        new_model.load(save_path)

        assert str(new_model.predict.signature) == str(model.predict.signature)
        assert new_model.predict.demos[0] == model.predict.demos[0].toDict()


def test_lm_dump_state_excludes_sensitive_keys():
    """Test that LM.dump_state() directly excludes sensitive keys."""
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key="sk-test-key",
        api_base="https://custom.com",
        base_url="https://another.com",
        temperature=0.7,
    )

    state = lm.dump_state()

    # Verify sensitive keys are excluded
    assert "api_key" not in state
    assert "api_base" not in state
    assert "base_url" not in state

    # Verify other keys are present
    assert state["model"] == "openai/gpt-4o-mini"
    assert state["temperature"] == 0.7
