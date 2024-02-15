import functools
import dsp
import dspy
from dspy.utils import DummyLM
from dspy import Signature, InputField, OutputField
from dspy.primitives.assertions import assert_transform_module, backtrack_handler


def test_retry_simple():
    predict = dspy.Predict("question -> answer")
    retry_module = dspy.Retry(predict)

    # Test Retry has created the correct new signature
    for field in predict.signature.output_fields:
        assert f"past_{field}" in retry_module.new_signature.input_fields
    assert "feedback" in retry_module.new_signature.input_fields

    dsp.settings.lm = lm = DummyLM(["blue"])
    result = retry_module.forward(
        question="What color is the sky?",
        past_outputs={"answer": "red"},
        feedback="Try harder",
    )
    assert result.answer == "blue"

    print(lm.get_convo(-1))
    assert lm.get_convo(-1).endswith(
        "Question: What color is the sky?\n\n"
        "Past Answer: red\n\n"
        "Instructions: Try harder\n\n"
        "Answer: blue"
    )


def test_retry_forward_with_feedback_suggest_success():
    # First we make a mistake, then we fix it
    dsp.settings.lm = lm = DummyLM(["red", "blue"])
    dsp.settings.trace = []

    class SimpleModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("question -> answer")

        def forward(self, **kwargs):
            result = self.predictor(**kwargs)
            print(f"SimpleModule got {result.answer=}")
            dspy.Suggest(result.answer == "blue", "Please think harder")
            return result

    program = SimpleModule()
    program = assert_transform_module(
        program.map_named_predictors(dspy.Retry),
        functools.partial(backtrack_handler, max_backtracks=1),
    )

    result = program(question="What color is the sky?")

    assert result.answer == "blue"

    print(lm.get_convo(-1))
    assert lm.get_convo(-1).endswith(
        "Question: What color is the sky?\n\n"
        "Past Answer: red\n\n"
        "Instructions: Please think harder\n\n"
        "Answer: blue"
    )

def test_retry_forward_with_feedback_assert_failure():
    # We make a mistake, but can't fix it
    dsp.settings.lm = lm = DummyLM(["red", "red"])
    dsp.settings.trace = []

    class SimpleModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("question -> answer")

        def forward(self, **kwargs):
            result = self.predictor(**kwargs)
            print(f"SimpleModule got {result.answer=}")
            try:
                dspy.Assert(result.answer == "blue", "Please think harder")
            except dspy.primitives.assertions.DSPyAssertionError as e:
                assert "Please think harder" in str(e)
            return result

    program = SimpleModule()
    program = assert_transform_module(
        program.map_named_predictors(dspy.Retry),
        functools.partial(backtrack_handler, max_backtracks=1),
    )

    result = program(question="What color is the sky?")

    assert result.answer == "red"