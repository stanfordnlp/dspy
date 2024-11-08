from enum import Enum
from typing import List

import pydantic

import dspy
from tests.reliability.utils import assert_program_output_correct, known_failing_models


def test_qa_with_pydantic_answer_model():
    class Answer(pydantic.BaseModel):
        value: str
        certainty: float = pydantic.Field(
            description="A value between 0 and 1 indicating the model's confidence in the answer."
        )
        comments: List[str] = pydantic.Field(
            description="At least two comments providing additional details about the answer."
        )

    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: Answer = dspy.OutputField()

    program = dspy.Predict(QA)
    answer = program(question="What is the capital of France?").answer

    assert_program_output_correct(
        program_output=answer.value,
        grading_guidelines="The answer should be Paris. Answer should not contain extraneous information.",
    )
    assert_program_output_correct(
        program_output=answer.comments, grading_guidelines="The comments should be relevant to the answer"
    )
    assert answer.certainty >= 0
    assert answer.certainty <= 1
    assert len(answer.comments) >= 2


def test_color_classification_using_enum():
    Color = Enum("Color", ["RED", "GREEN", "BLUE"])

    class Colorful(dspy.Signature):
        text: str = dspy.InputField()
        color: Color = dspy.OutputField()

    program = dspy.Predict(Colorful)
    color = program(text="The sky is blue").color

    assert color == Color.BLUE


def test_entity_extraction_with_multiple_primitive_outputs():
    class ExtractEntityFromDescriptionOutput(pydantic.BaseModel):
        entity_hu: str = pydantic.Field(description="The extracted entity in Hungarian, cleaned and lowercased.")
        entity_en: str = pydantic.Field(description="The English translation of the extracted Hungarian entity.")
        is_inverted: bool = pydantic.Field(
            description="Boolean flag indicating if the input is connected in an inverted way."
        )
        categories: str = pydantic.Field(description="English categories separated by '|' to which the entity belongs.")
        review: bool = pydantic.Field(
            description="Boolean flag indicating low confidence or uncertainty in the extraction."
        )

    class ExtractEntityFromDescription(dspy.Signature):
        """Extract an entity from a Hungarian description, provide its English translation, categories, and an inverted flag."""

        description: str = dspy.InputField(description="The input description in Hungarian.")
        entity: ExtractEntityFromDescriptionOutput = dspy.OutputField(
            description="The extracted entity and its properties."
        )

    program = dspy.ChainOfThought(ExtractEntityFromDescription)

    extracted_entity = program(description="A kávé egy növényi eredetű ital, amelyet a kávébabból készítenek.").entity
    assert_program_output_correct(
        program_output=extracted_entity.entity_hu,
        grading_guidelines="The translation of the text into English should be equivalent to 'coffee'",
    )
    assert_program_output_correct(
        program_output=extracted_entity.entity_hu,
        grading_guidelines="The text should be equivalent to 'coffee'",
    )
    assert_program_output_correct(
        program_output=extracted_entity.categories,
        grading_guidelines=(
            "The text should contain English language categories that apply to the word 'coffee'."
            " The categories should be separated by the character '|'."
        ),
    )
