from typing import Optional, Union

from pydantic import BaseModel, field_validator

import dspy


class SynthesizerArguments(BaseModel):
    # [TODO]
    feedback_mode: Optional[str] = None
    num_example_for_feedback: Optional[int] = None

    input_lm_model: Optional[dspy.LM] = None
    output_lm_model: Optional[dspy.LM] = None
    output_teacher_module: Optional[Union[dspy.Module, dspy.Predict]] = None

    num_example_for_optim: Optional[int] = None

    @field_validator(fields=["feedback_mode", "num_example_for_feedback"])
    def validate_feedback_mode(cls, value):
        if value and value not in ["human", "llm"]:
            raise ValueError("Feedback mode should be either 'human' or 'llm'.")
        
        if value and not cls.num_example_for_feedback:
            raise ValueError("Number of examples for feedback is required when feedback mode is provided.")

        return value