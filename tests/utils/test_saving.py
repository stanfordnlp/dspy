import logging
from unittest.mock import patch

import pytest

import dspy
from dspy.utils import DummyLM


def test_save_predict(tmp_path):
    predict = dspy.Predict("question->answer")
    predict.save(tmp_path, save_program=True)

    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "program.pkl").exists()

    loaded_predict = dspy.load(tmp_path, allow_pickle=True)
    assert isinstance(loaded_predict, dspy.Predict)

    assert predict.signature == loaded_predict.signature


def test_save_custom_model(tmp_path):
    class CustomModel(dspy.Module):
        def __init__(self):
            self.cot1 = dspy.ChainOfThought("question->refined_question")
            self.cot2 = dspy.ChainOfThought("refined_question->answer")

    model = CustomModel()
    model.save(tmp_path, save_program=True)

    loaded_model = dspy.load(tmp_path, allow_pickle=True)
    assert isinstance(loaded_model, CustomModel)

    assert len(model.predictors()) == len(loaded_model.predictors())
    for predictor, loaded_predictor in zip(model.predictors(), loaded_model.predictors(), strict=False):
        assert predictor.signature == loaded_predictor.signature


def test_save_model_with_custom_signature(tmp_path):
    import datetime

    class MySignature(dspy.Signature):
        """Just a custom signature."""

        current_date: datetime.date = dspy.InputField()
        target_date: datetime.date = dspy.InputField()
        date_diff: int = dspy.OutputField(desc="The difference in days between the current_date and the target_date")

    predict = dspy.Predict(MySignature)
    predict.signature = predict.signature.with_instructions("You are a helpful assistant.")
    predict.save(tmp_path, save_program=True)

    loaded_predict = dspy.load(tmp_path, allow_pickle=True)
    assert isinstance(loaded_predict, dspy.Predict)

    assert predict.signature == loaded_predict.signature


@pytest.mark.extra
def test_save_compiled_model(tmp_path):
    predict = dspy.Predict("question->answer")
    dspy.configure(lm=DummyLM([{"answer": "blue"}, {"answer": "white"}] * 10))

    trainset = [
        {"question": "What is the color of the sky?", "answer": "blue"},
        {"question": "What is the color of the ocean?", "answer": "blue"},
        {"question": "What is the color of the milk?", "answer": "white"},
        {"question": "What is the color of the coffee?", "answer": "black"},
    ]
    trainset = [dspy.Example(**example).with_inputs("question") for example in trainset]

    def dummy_metric(example, pred, trace=None):
        return True

    optimizer = dspy.BootstrapFewShot(max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5, metric=dummy_metric)
    compiled_predict = optimizer.compile(predict, trainset=trainset)
    compiled_predict.save(tmp_path, save_program=True)

    loaded_predict = dspy.load(tmp_path, allow_pickle=True)
    assert compiled_predict.demos == loaded_predict.demos
    assert compiled_predict.signature == loaded_predict.signature


def test_load_with_version_mismatch(tmp_path):
    from dspy.utils.saving import logger

    # Mock versions during save
    save_versions = {"python": "3.9", "dspy": "2.4.0", "cloudpickle": "2.0"}

    # Mock versions during load
    load_versions = {"python": "3.10", "dspy": "2.5.0", "cloudpickle": "2.1"}

    predict = dspy.Predict("question->answer")

    # Create a custom handler to capture log messages
    class ListHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    # Add handler and set level
    handler = ListHandler()
    original_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    try:
        # Mock version during save
        with patch("dspy.primitives.base_module.get_dependency_versions", return_value=save_versions):
            predict.save(tmp_path, save_program=True)

        # Mock version during load
        with patch("dspy.utils.saving.get_dependency_versions", return_value=load_versions):
            loaded_predict = dspy.load(tmp_path, allow_pickle=True)

        # Assert warnings were logged, and one warning for each mismatched dependency.
        assert len(handler.messages) == 3

        for msg in handler.messages:
            assert "There is a mismatch of" in msg

        # Verify the model still loads correctly despite version mismatches
        assert isinstance(loaded_predict, dspy.Predict)
        assert predict.signature == loaded_predict.signature

    finally:
        # Clean up: restore original level and remove handler
        logger.setLevel(original_level)
        logger.removeHandler(handler)


def test_pickle_loading_requires_explicit_permission(tmp_path):
    """Test that loading pickle files requires explicit permission."""
    predict = dspy.Predict("question->answer")
    predict.save(tmp_path, save_program=True)

    # Should fail without dangerously_allow_pickle
    with pytest.raises(ValueError, match="Loading with pickle is not allowed"):
        dspy.load(tmp_path)

    # Should succeed with dangerously_allow_pickle
    loaded_predict = dspy.load(tmp_path, allow_pickle=True)
    assert isinstance(loaded_predict, dspy.Predict)


def test_pkl_file_loading_requires_explicit_permission(tmp_path):
    """Test that loading .pkl files requires explicit permission."""
    predict = dspy.Predict("question->answer")
    pkl_path = tmp_path / "model.pkl"
    predict.save(pkl_path)

    # Should fail without allow_pickle
    new_predict = dspy.Predict("question->answer")
    with pytest.raises(ValueError, match="Loading .pkl files can run arbitrary code"):
        new_predict.load(pkl_path)

    # Should succeed with allow_pickle
    new_predict.load(pkl_path, allow_pickle=True)
    assert new_predict.dump_state() == predict.dump_state()


def test_json_file_loading_works_without_permission(tmp_path):
    """Test that loading .json files works without explicit permission."""
    predict = dspy.Predict("question->answer")
    json_path = tmp_path / "model.json"
    predict.save(json_path)

    # Should succeed without allow_pickle
    new_predict = dspy.Predict("question->answer")
    new_predict.load(json_path)
    assert new_predict.dump_state() == predict.dump_state()
