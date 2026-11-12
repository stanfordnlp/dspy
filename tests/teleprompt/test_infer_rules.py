import pytest

import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.teleprompt import InferRules


def answer_metric(example, prediction, trace=None):
    return example.answer == prediction.answer


trainset = [Example(question=f"train {i}", answer="good").with_inputs("question") for i in range(4)]


def test_induce_rules_drops_examples_until_prompt_fits_context_window():
    optimizer = InferRules(metric=answer_metric)
    prompts = []

    def rules_induction_program(examples_text):
        prompts.append(examples_text)
        if examples_text.count("Input Fields:") > 2:
            raise dspy.ContextWindowExceededError(model="dummy")
        return "RULE"

    optimizer.rules_induction_program = rules_induction_program

    rules = optimizer.induce_natural_language_rules(Predict("question -> answer"), trainset)

    assert rules == "RULE"
    assert [prompt.count("Input Fields:") for prompt in prompts] == [4, 3, 2]


def test_induce_rules_raises_when_a_single_example_exceeds_context_window():
    optimizer = InferRules(metric=answer_metric)

    def rules_induction_program(examples_text):
        raise dspy.ContextWindowExceededError(model="dummy")

    optimizer.rules_induction_program = rules_induction_program

    with pytest.raises(RuntimeError, match="single example") as exc_info:
        optimizer.induce_natural_language_rules(Predict("question -> answer"), trainset)

    assert isinstance(exc_info.value.__cause__, dspy.ContextWindowExceededError)
