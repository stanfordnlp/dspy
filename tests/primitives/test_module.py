import dspy
import threading
from dspy.utils.dummies import DummyLM
import logging
from unittest.mock import patch
import pytest
import os


def test_deepcopy_basic():
    signature = dspy.Signature("q -> a")
    cot = dspy.ChainOfThought(signature)
    cot_copy = cot.deepcopy()
    assert len(cot.parameters()) == len(cot_copy.parameters())
    # Parameters should be different objects with the same values.
    assert id(cot.parameters()[0]) != id(cot_copy.parameters()[0])
    assert cot.parameters()[0].__dict__ == cot_copy.parameters()[0].__dict__


def test_deepcopy_with_uncopyable_modules():
    class CustomClass(dspy.Module):
        def __init__(self):
            self.lock = threading.Lock()  # Non-copyable object.
            self.cot = dspy.ChainOfThought(dspy.Signature("q -> a"))

    model = CustomClass()
    model_copy = model.deepcopy()
    assert len(model.parameters()) == len(model_copy.parameters())
    # The lock should be refer to the same object (shallow copy).
    assert id(model.lock) == id(model_copy.lock)
    # Parameters should be different objects with the same values.
    assert id(model.parameters()[0]) != id(model_copy.parameters()[0])
    assert model.parameters()[0].__dict__ == model_copy.parameters()[0].__dict__


def test_deepcopy_with_nested_modules():
    class CustomClass1(dspy.Module):
        def __init__(self):
            self.lock = threading.Lock()  # Non-copyable object.
            self.cot = dspy.ChainOfThought(dspy.Signature("q -> a"))

    class CustomClass2(dspy.Module):
        def __init__(self):
            self.submodel = CustomClass1()

    model = CustomClass2()
    model_copy = model.deepcopy()
    assert len(model.parameters()) == len(model_copy.parameters())
    # The lock should be refer to the same object (shallow copy).
    assert id(model.submodel.lock) == id(model_copy.submodel.lock)
    # Parameters should be different objects with the same values.
    assert id(model.parameters()[0]) != id(model_copy.parameters()[0])
    assert model.parameters()[0].__dict__ == model_copy.parameters()[0].__dict__


def test_save_and_load_with_json(tmp_path):
    model = dspy.ChainOfThought(dspy.Signature("q -> a"))
    model.predict.signature = model.predict.signature.with_instructions("You are a helpful assistant.")
    model.predict.demos = [
        dspy.Example(q="What is the capital of France?", a="Paris", reasoning="n/a").with_inputs("q", "a")
    ]
    save_path = tmp_path / "model.json"
    model.save(save_path)
    new_model = dspy.ChainOfThought(dspy.Signature("q -> a"))
    new_model.load(save_path)

    assert str(new_model.predict.signature) == str(model.predict.signature)
    assert new_model.predict.demos[0] == model.predict.demos[0].toDict()


def test_save_and_load_with_pkl(tmp_path):
    import datetime

    # `datetime.date` is not json serializable, so we need to save with pickle.
    class MySignature(dspy.Signature):
        """Just a custom signature."""

        current_date: datetime.date = dspy.InputField()
        target_date: datetime.date = dspy.InputField()
        date_diff: int = dspy.OutputField(desc="The difference in days between the current_date and the target_date")

    trainset = [
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 2), "date_diff": 1},
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 3), "date_diff": 2},
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 4), "date_diff": 3},
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 5), "date_diff": 4},
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 6), "date_diff": 5},
    ]
    trainset = [dspy.Example(**example).with_inputs("current_date", "target_date") for example in trainset]

    dspy.settings.configure(
        lm=DummyLM([{"date_diff": "1", "reasoning": "n/a"}, {"date_diff": "2", "reasoning": "n/a"}] * 10)
    )

    cot = dspy.ChainOfThought(MySignature)
    cot(current_date=datetime.date(2024, 1, 1), target_date=datetime.date(2024, 1, 2))

    def dummy_metric(example, pred, trace=None):
        return True

    optimizer = dspy.BootstrapFewShot(max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5, metric=dummy_metric)
    compiled_cot = optimizer.compile(cot, trainset=trainset)
    compiled_cot.predict.signature = compiled_cot.predict.signature.with_instructions("You are a helpful assistant.")

    save_path = tmp_path / "program.pkl"
    compiled_cot.save(save_path)

    new_cot = dspy.ChainOfThought(MySignature)
    new_cot.load(save_path)

    assert str(new_cot.predict.signature) == str(compiled_cot.predict.signature)
    assert new_cot.predict.demos == compiled_cot.predict.demos


def test_load_with_version_mismatch(tmp_path):
    from dspy.primitives.module import logger

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
        save_path = tmp_path / "program.pkl"
        # Mock version during save
        with patch("dspy.primitives.module.get_dependency_versions", return_value=save_versions):
            predict.save(save_path)

        # Mock version during load
        with patch("dspy.primitives.module.get_dependency_versions", return_value=load_versions):
            loaded_predict = dspy.Predict("question->answer")
            loaded_predict.load(save_path)

        # Assert warnings were logged, and one warning for each mismatched dependency.
        assert len(handler.messages) == 3

        for msg in handler.messages:
            assert "There is a mismatch of" in msg

        # Verify the model still loads correctly despite version mismatches
        assert isinstance(loaded_predict, dspy.Predict)
        assert str(predict.signature) == str(loaded_predict.signature)

    finally:
        # Clean up: restore original level and remove handler
        logger.setLevel(original_level)
        logger.removeHandler(handler)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skip the test if OPENAI_API_KEY is not set.")
def test_single_module_call_with_usage_tracker():
    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False), track_usage=True)

    predict = dspy.ChainOfThought("question -> answer")
    output = predict(question="What is the capital of France?")

    lm_usage = output.get_lm_usage()
    assert len(lm_usage) == 1
    assert lm_usage["openai/gpt-4o-mini"]["prompt_tokens"] > 0
    assert lm_usage["openai/gpt-4o-mini"]["completion_tokens"] > 0
    assert lm_usage["openai/gpt-4o-mini"]["total_tokens"] > 0

    # Test no usage being tracked when cache is enabled
    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=True), track_usage=True)
    for _ in range(2):
        output = predict(question="What is the capital of France?")

    assert len(output.get_lm_usage()) == 0


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skip the test if OPENAI_API_KEY is not set.")
def test_multi_module_call_with_usage_tracker():
    dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False), track_usage=True)

    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict1 = dspy.ChainOfThought("question -> answer")
            self.predict2 = dspy.ChainOfThought("question, answer -> score")

        def __call__(self, question: str) -> str:
            answer = self.predict1(question=question)
            score = self.predict2(question=question, answer=answer)
            return score

    program = MyProgram()
    output = program(question="What is the capital of France?")

    lm_usage = output.get_lm_usage()
    assert len(lm_usage) == 1
    assert lm_usage["openai/gpt-4o-mini"]["prompt_tokens"] > 0
    assert lm_usage["openai/gpt-4o-mini"]["prompt_tokens"] > 0
    assert lm_usage["openai/gpt-4o-mini"]["completion_tokens"] > 0
    assert lm_usage["openai/gpt-4o-mini"]["total_tokens"] > 0


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skip the test if OPENAI_API_KEY is not set.")
def test_usage_tracker_in_parallel():
    class MyProgram(dspy.Module):
        def __init__(self, lm):
            self.lm = lm
            self.predict1 = dspy.ChainOfThought("question -> answer")
            self.predict2 = dspy.ChainOfThought("question, answer -> score")

        def __call__(self, question: str) -> str:
            with dspy.settings.context(lm=self.lm):
                answer = self.predict1(question=question)
                score = self.predict2(question=question, answer=answer)
                return score

    dspy.settings.configure(track_usage=True)
    program1 = MyProgram(lm=dspy.LM("openai/gpt-4o-mini", cache=False))
    program2 = MyProgram(lm=dspy.LM("openai/gpt-3.5-turbo", cache=False))

    parallelizer = dspy.Parallel()

    results = parallelizer(
        [
            (program1, {"question": "What is the meaning of life?"}),
            (program2, {"question": "why did a chicken cross the kitchen?"}),
        ]
    )

    assert results[0].get_lm_usage() is not None
    assert results[1].get_lm_usage() is not None

    assert results[0].get_lm_usage().keys() == set(["openai/gpt-4o-mini"])
    assert results[1].get_lm_usage().keys() == set(["openai/gpt-3.5-turbo"])
