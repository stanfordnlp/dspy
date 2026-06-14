"""
Question Answering DSPy program definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy

from programs.base import BaseProgram, format_context

if TYPE_CHECKING:
    from dspy import Prediction


class NaiveQA(dspy.Module, BaseProgram):
    """Naive QA program that directly predicts the answer from the question."""

    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict(
            dspy.Signature(
                "question -> answer",
            )
        )

    def forward(self, question: str, **kwargs) -> Prediction:
        return self.answer(question=question)
    
    @property
    def name(self) -> str:
        return "naive"


class ReasoningFirstQA(dspy.Module, BaseProgram):
    """Multi-step QA program with separate think and answer predictors.

    This program uses a two-stage approach:
    1. Think: Analyze the question and generate reasoning
    2. Answer: Use the reasoning to produce a concise answer

    GEPA and other optimizers can optimize both predictor instructions independently.
    """

    def __init__(self):
        super().__init__()
        self.think = dspy.ChainOfThought(
            dspy.Signature(
                "question -> reasoning",
            )
        )
        self.answer = dspy.Predict(
            dspy.Signature(
                "question, reasoning -> answer",
            )
        )

    def forward(self, question: str, **kwargs) -> Prediction:
        think_result = self.think(question=question)
        return self.answer(question=question, reasoning=think_result.reasoning)
    
    @property
    def name(self) -> str:
        return "reasoning"


class ContextQA(dspy.Module, BaseProgram):
    """QA program that uses provided context to answer questions.
    
    This program takes both a question and context (e.g., retrieved passages)
    and uses them together to generate an answer.
    
    Supports HotPotQA dict format, list of strings, or a single string.
    """

    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict(
            dspy.Signature(
                "context, question -> answer",
            )
        )

    def forward(self, question: str, context: dict | list[str] | str = None, **kwargs) -> Prediction:
        context_str = format_context(context)
        return self.answer(context=context_str, question=question)
    
    @property
    def name(self) -> str:
        return "context"


class ReasoningContextQA(dspy.Module, BaseProgram):
    """Multi-step QA program that uses context with reasoning.

    This program uses a two-stage approach with context:
    1. Think: Analyze the context and question to generate reasoning
    2. Answer: Use the reasoning to produce a concise answer
    
    Supports HotPotQA dict format, list of strings, or a single string.
    """

    def __init__(self):
        super().__init__()
        self.think = dspy.ChainOfThought(
            dspy.Signature(
                "context, question -> reasoning",
            )
        )
        self.answer = dspy.Predict(
            dspy.Signature(
                "context, question, reasoning -> answer",
            )
        )

    def forward(self, question: str, context: dict | list[str] | str = None, **kwargs) -> Prediction:
        context_str = format_context(context)
        think_result = self.think(context=context_str, question=question)
        return self.answer(context=context_str, question=question, reasoning=think_result.reasoning)
    
    @property
    def name(self) -> str:
        return "reasoning_context"


# Define signature at program level (Option 1: Program-level class)
class ContextQASignatureV1(dspy.Signature):
    """You are a question answering assistant. Answer questions based ONLY on the provided context.

IMPORTANT INSTRUCTIONS:
- For yes/no questions, answer ONLY "yes" or "no"
- Do NOT include phrases like "based on the context" or "according to the documents"
"""
    context: str = dspy.InputField(desc="The context to use for answering the question")
    question: str = dspy.InputField(desc="The question to answer")
    answer: str = dspy.OutputField(desc="The answer to the question")


class MLflowBasePromptContextQA(dspy.Module, BaseProgram):
    """QA program with a structured base prompt for context-based question answering.
    
    Uses program-level signature class definition.
    """

    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict(ContextQASignatureV1)

    def forward(self, question: str, context: dict | list[str] | str = None, **kwargs) -> Prediction:
        context_str = format_context(context)
        return self.answer(context=context_str, question=question)
    
    @property
    def name(self) -> str:
        return "mlflow_base_prompt"


# Option 2: Using string signature with .with_instructions()
class MLflowBasePromptContextQAv2(dspy.Module, BaseProgram):
    """QA program using string signature with .with_instructions() method."""

    INSTRUCTION = """You are a question answering assistant. Answer questions based ONLY on the provided context.

IMPORTANT INSTRUCTIONS:
- For yes/no questions, answer ONLY "yes" or "no"
- Do NOT include phrases like "based on the context" or "according to the documents"
"""

    def __init__(self):
        super().__init__()
        # Create signature from string, then add instructions
        self.answer = dspy.Predict(dspy.Signature(
            "context, question -> answer",
            self.INSTRUCTION
        ))

    def forward(self, question: str, context: dict | list[str] | str = None, **kwargs) -> Prediction:
        context_str = format_context(context)
        return self.answer(context=context_str, question=question)
    
    @property
    def name(self) -> str:
        return "mlflow_v2"


# Option 3: Simple string signature (let GEPA propose the instruction from scratch)
class SimpleContextQA(dspy.Module, BaseProgram):
    """QA program with minimal signature - lets GEPA propose instructions from scratch."""

    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict("context, question -> answer")

    def forward(self, question: str, context: dict | list[str] | str = None, **kwargs) -> Prediction:
        context_str = format_context(context)
        return self.answer(context=context_str, question=question)

    @property
    def name(self) -> str:
        return "simple_context"


class MathNaive(dspy.Module, BaseProgram):
    """Math problem solving program that directly predicts the answer from the problem."""

    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict(
            dspy.Signature(
                "problem -> solution, answer",
                "Solve the given math problem. Present your solution concisely and then the final answer."
            )
        )

    def forward(self, problem: str, **kwargs) -> Prediction:
        return self.answer(problem=problem)

    @property
    def name(self) -> str:
        return "math_naive"


class MathAnswerOnly(dspy.Module, BaseProgram):
    """Math problem solving program that predicts only the final AIME answer."""

    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict(
            dspy.Signature(
                "problem -> answer",
                (
                    "Solve the given math problem internally. Return only the final numerical answer as an integer. "
                    "Do not include reasoning, units, markdown, or any additional text."
                ),
            )
        )

    def forward(self, problem: str, **kwargs) -> Prediction:
        return self.answer(problem=problem)

    @property
    def name(self) -> str:
        return "math_answer_only"
    
class MathCoT(dspy.Module, BaseProgram):
    """Math problem solving program using Chain of Thought.

    This program is designed for mathematical reasoning tasks like AIME.
    It uses ChainOfThought to generate step-by-step reasoning before
    producing the final numerical answer.
    """

    def __init__(self):
        super().__init__()
        self.solve = dspy.ChainOfThought(
            dspy.Signature(
                "problem -> answer",
                "Solve the given math problem. Your answer must be ONLY the final numerical result as an integer (e.g., 42, 157, 1000). Do not include any additional text, explanation, units, or formatting."
            ),
            rationale_field=dspy.OutputField(
                prefix="Reasoning:",
                desc="Solve the problem within less than 1000 words",
            ),
            rationale_field_type=str,
        )

    def forward(self, problem: str, **kwargs) -> Prediction:
        """Solve a math problem with chain-of-thought reasoning.

        Args:
            problem: The math problem to solve.

        Returns:
            Prediction with reasoning and answer fields.
        """
        return self.solve(problem=problem)

    @property
    def name(self) -> str:
        return "math_cot"
