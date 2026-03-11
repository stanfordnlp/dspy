"""Small demo script for Predict call ergonomics.

Use this in VS Code or from the terminal to inspect the new call surface.

Suggested usage:

    python docs/predict_ergonomics_demo.py

Then also open this file in VS Code and hover over `predict`,
`inspect.signature(predict)`, and the call sites below.
"""

from __future__ import annotations

import inspect
import logging

import dspy


class DummyPredictLM(dspy.LM):
    """Tiny LM stub for local experimentation."""

    def __init__(self):
        super().__init__("dummy")
        self.calls = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        self.calls.append({"prompt": prompt, "messages": messages, "kwargs": kwargs})
        return ["[[ ## answer ## ]]\nParis"]


class QA(dspy.Signature):
    """Answer a question using the supplied context."""

    question: str = dspy.InputField()
    context: list[str] = dspy.InputField()
    answer: str = dspy.OutputField()


class UntypedQuestion(dspy.Signature):
    """Answer a question with no explicit input type hint."""

    question = dspy.InputField()
    answer: str = dspy.OutputField()


class WithDefault(dspy.Signature):
    """Answer a question with optional context."""

    context: str = dspy.InputField(default="DEFAULT_CONTEXT")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def print_header(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    lm = DummyPredictLM()
    dspy.configure(lm=lm)

    print_header("Typed signature")
    predict = dspy.Predict(QA)
    print("inspect.signature(predict):")
    print(inspect.signature(predict))

    print("\nPositional call:")
    result = predict("What is the capital of France?", ["Use world knowledge."])
    print(result)

    print("\nMixed positional/keyword call:")
    result = predict("What is the capital of France?", context=["Be concise."])
    print(result)

    print("\nCall-time signature override:")
    concise = predict.signature.with_instructions("Answer in one short sentence.")
    result = predict(
        question="Why is the sky blue?",
        context=["Use world knowledge."],
        signature_override=concise,
        config={"temperature": 0.2},
    )
    print(result)

    print_header("Untyped signature")
    untyped = dspy.Predict(UntypedQuestion)
    print("inspect.signature(untyped):")
    print(inspect.signature(untyped))
    print("\nCalling untyped predictor with an int question (should not warn about type mismatch):")
    print(untyped(question=123))

    print_header("Default-before-required edge case")
    with_default = dspy.Predict(WithDefault)
    print("inspect.signature(with_default):")
    print(inspect.signature(with_default))
    print("\nKeyword call using defaulted field:")
    print(with_default(question="What is the capital of France?"))

    print_header("Recent LM call payload")
    if lm.calls:
        print(lm.calls[-1])


if __name__ == "__main__":
    main()
