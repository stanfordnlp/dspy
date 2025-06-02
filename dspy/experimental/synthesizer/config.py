from typing import Any, Optional

from pydantic import BaseModel, model_validator


class SynthesizerArguments(BaseModel):
    feedback_mode: Optional[str] = None
    num_example_for_feedback: Optional[int] = None

    input_lm_model: Optional[Any] = None
    output_lm_model: Optional[Any] = None
    output_teacher_module: Optional[Any] = None

    num_example_for_optim: Optional[int] = None

    @model_validator(mode="after")
    def validate_feedback_mode(self):
        if self.feedback_mode and self.feedback_mode not in ["human", "llm"]:
            raise ValueError("Feedback mode should be either 'human' or 'llm'.")

        if self.feedback_mode and not self.num_example_for_feedback:
            raise ValueError("Number of examples for feedback is required when feedback mode is provided.")

        return self
