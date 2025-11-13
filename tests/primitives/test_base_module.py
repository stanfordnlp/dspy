import asyncio
import logging
import os
import threading
from unittest.mock import patch

import pytest
from litellm import Choices, Message, ModelResponse
from litellm.types.utils import Usage

import dspy
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM


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
        dspy.Example(q="What is the capital of France?", a="Paris", reasoning="n/a").with_inputs("q"),
        # Nested example
        dspy.Example(
            q=[
                dspy.Example(q="What is the capital of France?"),
                dspy.Example(q="What is actually the capital of France?"),
            ],
            a="Paris",
            reasoning="n/a",
        ).with_inputs("q"),
    ]
    save_path = tmp_path / "model.json"
    model.save(save_path)
    new_model = dspy.ChainOfThought(dspy.Signature("q -> a"))
    new_model.load(save_path)

    assert str(new_model.predict.signature) == str(model.predict.signature)
    assert new_model.predict.demos[0] == model.predict.demos[0].toDict()
    assert new_model.predict.demos[1] == model.predict.demos[1].toDict()


@pytest.mark.extra
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

    dspy.configure(
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
    new_cot.load(save_path, allow_pickle=True)

    assert str(new_cot.predict.signature) == str(compiled_cot.predict.signature)
    assert new_cot.predict.demos == compiled_cot.predict.demos


def test_save_with_extra_modules(tmp_path):
    import sys

    # Create a temporary Python file with our custom module
    custom_module_path = tmp_path / "custom_module.py"
    with open(custom_module_path, "w") as f:
        f.write("""
import dspy

class MyModule(dspy.Module):
    def __init__(self):
        self.cot = dspy.ChainOfThought(dspy.Signature("q -> a"))

    def forward(self, q):
        return self.cot(q=q)
""")

    # Add the tmp_path to Python path so we can import the module
    sys.path.insert(0, str(tmp_path))
    try:
        import custom_module

        cot = custom_module.MyModule()

        cot.save(tmp_path, save_program=True)
        # Remove the custom module from sys.modules to simulate it not being available
        sys.modules.pop("custom_module", None)
        # Also remove it from sys.path
        sys.path.remove(str(tmp_path))
        del custom_module

        # Test the loading fails without using `modules_to_serialize`
        with pytest.raises(ModuleNotFoundError):
            dspy.load(tmp_path, allow_pickle=True)

        sys.path.insert(0, str(tmp_path))
        import custom_module

        cot.save(
            tmp_path,
            modules_to_serialize=[custom_module],
            save_program=True,
        )

        # Remove the custom module from sys.modules to simulate it not being available
        sys.modules.pop("custom_module", None)
        # Also remove it from sys.path
        sys.path.remove(str(tmp_path))
        del custom_module

        loaded_module = dspy.load(tmp_path, allow_pickle=True)
        assert loaded_module.cot.predict.signature == cot.cot.predict.signature

    finally:
        # Only need to clean up sys.path
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))


def test_load_with_version_mismatch(tmp_path):
    from dspy.primitives.base_module import logger

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
        with patch("dspy.primitives.base_module.get_dependency_versions", return_value=save_versions):
            predict.save(save_path)

        # Mock version during load
        with patch("dspy.primitives.base_module.get_dependency_versions", return_value=load_versions):
            loaded_predict = dspy.Predict("question->answer")
            loaded_predict.load(save_path, allow_pickle=True)

        # Assert warnings were logged: 1 for pickle loading + 3 for version mismatches
        assert len(handler.messages) == 4

        # First message is about pickle loading
        assert ".pkl" in handler.messages[0]

        # Rest are version mismatch warnings
        for msg in handler.messages[1:]:
            assert "There is a mismatch of" in msg

        # Verify the model still loads correctly despite version mismatches
        assert isinstance(loaded_predict, dspy.Predict)
        assert str(predict.signature) == str(loaded_predict.signature)

    finally:
        # Clean up: restore original level and remove handler
        logger.setLevel(original_level)
        logger.removeHandler(handler)


@pytest.mark.llm_call
def test_single_module_call_with_usage_tracker(lm_for_test):
    dspy.configure(lm=dspy.LM(lm_for_test, cache=False), track_usage=True)

    predict = dspy.ChainOfThought("question -> answer")
    output = predict(question="What is the capital of France?")

    lm_usage = output.get_lm_usage()
    assert len(lm_usage) == 1
    assert lm_usage[lm_for_test]["prompt_tokens"] > 0
    assert lm_usage[lm_for_test]["completion_tokens"] > 0
    assert lm_usage[lm_for_test]["total_tokens"] > 0

    # Test no usage being tracked when cache is enabled
    dspy.configure(lm=dspy.LM(lm_for_test, cache=True), track_usage=True)
    for _ in range(2):
        output = predict(question="What is the capital of France?")

    assert len(output.get_lm_usage()) == 0


@pytest.mark.llm_call
def test_multi_module_call_with_usage_tracker(lm_for_test):
    dspy.configure(lm=dspy.LM(lm_for_test, cache=False), track_usage=True)

    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict1 = dspy.ChainOfThought("question -> answer")
            self.predict2 = dspy.ChainOfThought("question, answer -> score")

        def __call__(self, question: str) -> Prediction:
            answer = self.predict1(question=question)
            score = self.predict2(question=question, answer=answer)
            return score

    program = MyProgram()
    output = program(question="What is the capital of France?")

    lm_usage = output.get_lm_usage()
    assert len(lm_usage) == 1
    assert lm_usage[lm_for_test]["prompt_tokens"] > 0
    assert lm_usage[lm_for_test]["prompt_tokens"] > 0
    assert lm_usage[lm_for_test]["completion_tokens"] > 0
    assert lm_usage[lm_for_test]["total_tokens"] > 0


# TODO: prepare second model for testing this unit test in ci
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skip the test if OPENAI_API_KEY is not set.")
def test_usage_tracker_in_parallel():
    class MyProgram(dspy.Module):
        def __init__(self, lm):
            self.lm = lm
            self.predict1 = dspy.ChainOfThought("question -> answer")
            self.predict2 = dspy.ChainOfThought("question, answer -> score")

        def __call__(self, question: str) -> Prediction:
            with dspy.context(lm=self.lm):
                answer = self.predict1(question=question)
                score = self.predict2(question=question, answer=answer)
                return score

    dspy.configure(track_usage=True)
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


@pytest.mark.asyncio
async def test_usage_tracker_async_parallel():
    program = dspy.Predict("question -> answer")

    with patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'answer': 'Paris'}"))],
            usage=Usage(
                **{
                    "prompt_tokens": 1117,
                    "completion_tokens": 46,
                    "total_tokens": 1163,
                    "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                    "completion_tokens_details": {
                        "reasoning_tokens": 0,
                        "audio_tokens": 0,
                        "accepted_prediction_tokens": 0,
                        "rejected_prediction_tokens": 0,
                    },
                },
            ),
            model="openai/gpt-4o-mini",
        )

        coroutines = [
            program.acall(question="What is the capital of France?"),
            program.acall(question="What is the capital of France?"),
            program.acall(question="What is the capital of France?"),
            program.acall(question="What is the capital of France?"),
        ]
        with dspy.context(
            lm=dspy.LM("openai/gpt-4o-mini", cache=False), track_usage=True, adapter=dspy.JSONAdapter()
        ):
            results = await asyncio.gather(*coroutines)

        assert results[0].get_lm_usage() is not None
        assert results[1].get_lm_usage() is not None

        lm_usage0 = results[0].get_lm_usage()["openai/gpt-4o-mini"]
        lm_usage1 = results[1].get_lm_usage()["openai/gpt-4o-mini"]
        assert lm_usage0["prompt_tokens"] == 1117
        assert lm_usage1["prompt_tokens"] == 1117
        assert lm_usage0["completion_tokens"] == 46
        assert lm_usage1["completion_tokens"] == 46
        assert lm_usage0["total_tokens"] == 1163
        assert lm_usage1["total_tokens"] == 1163


def test_usage_tracker_no_side_effect():
    class MyProgram(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict("question -> answer")

        def forward(self, question: str, **kwargs) -> str:
            return self.predict(question=question).answer

    program = MyProgram()
    with dspy.context(lm=DummyLM([{"answer": "Paris"}]), track_usage=True):
        result = program(question="What is the capital of France?")
    assert result == "Paris"


def test_module_history():
    class MyProgram(dspy.Module):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.cot = dspy.ChainOfThought("question -> answer")

        def forward(self, question: str, **kwargs) -> Prediction:
            return self.cot(question=question)

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(message=Message(content="{'reasoning': 'Paris is the capital of France', 'answer': 'Paris'}"))
            ],
            model="openai/gpt-4o-mini",
        )
        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter())
        program = MyProgram()
        program(question="What is the capital of France?")

        # Second call only call the submodule.
        program.cot(question="What is the capital of France?")

        # The LM history entity exists in all the ancestor callers.
        assert len(program.history) == 1
        assert len(program.cot.history) == 2
        assert len(program.cot.predict.history) == 2

        # The same history entity is shared across all the ancestor callers to reduce memory usage.
        assert id(program.history[0]) == id(program.cot.history[0])

        assert program.history[0]["outputs"] == ["{'reasoning': 'Paris is the capital of France', 'answer': 'Paris'}"]

        dspy.configure(disable_history=True)

        program(question="What is the capital of France?")
        # No history is recorded when history is disabled.
        assert len(program.history) == 1
        assert len(program.cot.history) == 2
        assert len(program.cot.predict.history) == 2

        dspy.configure(disable_history=False)

        program(question="What is the capital of France?")
        # History is recorded again when history is enabled.
        assert len(program.history) == 2
        assert len(program.cot.history) == 3
        assert len(program.cot.predict.history) == 3


def test_module_history_with_concurrency():
    class MyProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.cot = dspy.ChainOfThought("question -> answer")

        def forward(self, question: str, **kwargs) -> Prediction:
            return self.cot(question=question)

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="{'reasoning': 'N/A', 'answer': 'Holy crab!'}"))],
            model="openai/gpt-4o-mini",
        )
        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter())
        program = MyProgram()

        parallelizer = dspy.Parallel()

        parallelizer(
            [
                (program, {"question": "What is the meaning of life?"}),
                (program, {"question": "why did a chicken cross the kitchen?"}),
            ]
        )
        assert len(program.history) == 2
        assert len(program.cot.history) == 2
        assert len(program.cot.predict.history) == 2


@pytest.mark.asyncio
async def test_module_history_async():
    class MyProgram(dspy.Module):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.cot = dspy.ChainOfThought("question -> answer")

        async def aforward(self, question: str, **kwargs) -> Prediction:
            return await self.cot.acall(question=question)

    with patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[
                Choices(message=Message(content="{'reasoning': 'Paris is the capital of France', 'answer': 'Paris'}"))
            ],
            model="openai/gpt-4o-mini",
        )
        program = MyProgram()
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter()):
            await program.acall(question="What is the capital of France?")

            # Second call only call the submodule.
            await program.cot.acall(question="What is the capital of France?")

        # The LM history entity exists in all the ancestor callers.
        assert len(program.history) == 1
        assert len(program.cot.history) == 2
        assert len(program.cot.predict.history) == 2

        # The same history entity is shared across all the ancestor callers to reduce memory usage.
        assert id(program.history[0]) == id(program.cot.history[0])

        assert program.history[0]["outputs"] == ["{'reasoning': 'Paris is the capital of France', 'answer': 'Paris'}"]

        with dspy.context(
            disable_history=True, lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter()
        ):
            await program.acall(question="What is the capital of France?")

        # No history is recorded when history is disabled.
        assert len(program.history) == 1
        assert len(program.cot.history) == 2
        assert len(program.cot.predict.history) == 2

        with dspy.context(
            disable_history=False, lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=dspy.JSONAdapter()
        ):
            await program.acall(question="What is the capital of France?")
        # History is recorded again when history is enabled.
        assert len(program.history) == 2
        assert len(program.cot.history) == 3
        assert len(program.cot.predict.history) == 3


def test_forward_direct_call_warning(capsys):
    class TestModule(dspy.Module):
        def forward(self, x):
            return x

    module = TestModule()
    module.forward("test")
    captured = capsys.readouterr()
    assert "directly is discouraged" in captured.err


def test_forward_through_call_no_warning(capsys):
    class TestModule(dspy.Module):
        def forward(self, x):
            return x

    module = TestModule()
    module(x="test")
    captured = capsys.readouterr()
    assert "directly is discouraged" not in captured.err
