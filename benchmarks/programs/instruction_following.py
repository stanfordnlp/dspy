"""
Instruction-following DSPy program definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy

from programs.base import BaseProgram

if TYPE_CHECKING:
    from dspy import Prediction


class InstructionFollowing(dspy.Module, BaseProgram):
    """Single-predictor program for verifiable instruction following."""

    DEFAULT_INSTRUCTION = (
        "Follow every instruction in the user prompt exactly. "
        "Satisfy all explicit constraints such as length, format, keywords, and structure. "
        "Return only the requested content with no preamble or explanation unless asked."
    )

    def __init__(self):
        super().__init__()
        self.respond = dspy.Predict(
            dspy.Signature(
                "prompt -> response",
                self.DEFAULT_INSTRUCTION,
            )
        )

    def forward(self, prompt: str, **kwargs) -> Prediction:
        return self.respond(prompt=prompt)

    @property
    def name(self) -> str:
        return "instruction_following"
