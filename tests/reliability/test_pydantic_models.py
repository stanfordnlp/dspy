from enum import Enum
from typing import Any, List, Literal

import pydantic
import pytest

import dspy
from tests.reliability.utils import assert_program_output_correct, known_failing_models

@pytest.mark.reliability
def test_qa_with_pydantic_answer_model():
    class Answer(pydantic.BaseModel):
        value: str
        certainty: float = pydantic.Field(
            description="A value between 0 and 1 indicating the model's confidence in the answer."
        )
        comments: list[str] = pydantic.Field(
            description="At least two comments providing additional details about the answer."
        )

    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: Answer = dspy.OutputField()

    program = dspy.Predict(QA)
    question = "What is the capital of France?"
    answer = program(question=question).answer

    assert_program_output_correct(
        program_input=question,
        program_output=answer.value,
        grading_guidelines="The answer should be Paris. Answer should not contain extraneous information.",
    )
    assert_program_output_correct(
        program_input=question,
        program_output=answer.comments,
        grading_guidelines=(
            "The comments should be relevant to the answer. They don't need to restate the answer explicitly."
        ),
    )
    assert answer.certainty >= 0
    assert answer.certainty <= 1
    assert len(answer.comments) >= 2


@pytest.mark.parametrize("module", [dspy.Predict, dspy.ChainOfThought])
@pytest.mark.reliability
def test_color_classification_using_enum(module):
    Color = Enum("Color", ["RED", "GREEN", "BLUE"])

    class Colorful(dspy.Signature):
        text: str = dspy.InputField()
        color: Color = dspy.OutputField()

    program = module(Colorful)
    # Note: The precise text, including the trailing period, is important here for ensuring that
    # the program is correctly extracting the color from the text; previous implementations have
    # produced invalid enum responses for "The sky is blue.", but they have produced valid enum
    # responses for "The sky is blue" (without the period).
    color = program(text="The sky is blue.").color

    assert color == Color.BLUE


@pytest.mark.reliability
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
    description = "A kávé egy növényi eredetű ital, amelyet a kávébabból készítenek."

    extracted_entity = program(description=description).entity
    assert_program_output_correct(
        program_input=description,
        program_output=extracted_entity.entity_hu,
        grading_guidelines="The translation of the extracted entity into English should be equivalent to 'coffee'",
    )
    assert_program_output_correct(
        program_input=description,
        program_output=extracted_entity.entity_en,
        grading_guidelines="The extracted entity should be equivalent to 'coffee'",
    )
    assert_program_output_correct(
        program_input=description,
        program_output=extracted_entity.categories,
        grading_guidelines=(
            "The extracted entity should be associated with English language categories that apply to the word 'coffee'."
            " The categories should be separated by the character '|'."
        ),
    )


@pytest.mark.parametrize("module", [dspy.Predict, dspy.ChainOfThought])
@pytest.mark.reliability
def test_tool_calling_with_literals(module):
    next_tool_names = [
        "get_docs",
        "finish",
        "search_policy",
        "notify_manager",
        "calculate_accrual",
        "combine_leave",
        "review_seniority_rules",
        "fetch_calendar",
        "verify_compensation",
        "check_carryover_policy",
    ]

    class ToolCalling(dspy.Signature):
        """
        Given the fields question, produce the fields response.
        You will be given question and your goal is to finish with response.
        To do this, you will interleave Thought, Tool Name, and Tool Args, and receive a resulting Observation.
        """

        question: str = dspy.InputField()
        trajectory: str = dspy.InputField()
        next_thought: str = dspy.OutputField()
        next_tool_name: Literal[
            "get_docs",
            "finish",
            "search_policy",
            "notify_manager",
            "calculate_accrual",
            "combine_leave",
            "review_seniority_rules",
            "fetch_calendar",
            "verify_compensation",
            "check_carryover_policy",
        ] = dspy.OutputField()
        next_tool_args: dict[str, Any] = dspy.OutputField()
        response_status: Literal["success", "error", "pending"] = dspy.OutputField()
        user_intent: Literal["informational", "transactional", "exploratory"] = dspy.OutputField()

    program = dspy.Predict(ToolCalling)
    prediction = program(
        question=(
            "Tell me more about the company's internal policy for paid time off (PTO), "
            "including as many details as possible. I want to know how PTO is accrued—are "
            "there fixed rates, and do they vary by employee seniority or length of service? "
            "Are there specific rules about carrying unused PTO into the next calendar year, "
            "or is it a 'use it or lose it' system? Additionally, if an employee plans to take "
            "extended leave for a vacation or personal reasons, what is the process for submitting "
            "a request, and how far in advance should they notify their manager? Is there any overlap "
            "or interaction between PTO and other forms of leave, such as sick leave or parental leave? "
            "For example, can PTO be combined with those leave types to create a longer absence, or are "
            "they strictly separate? I’d also like to know if there are any restrictions on when PTO can "
            "be used, such as during critical business periods or holidays. Finally, what is the official "
            "policy if an employee resigns or is terminated—are they compensated for unused PTO days, and if "
            "so, at what rate?"
        ),
        trajectory=(
            "["
            "{'thought': 'Start by understanding PTO accrual rules.', 'tool_name': 'search_policy', 'tool_args': {'topic': 'PTO accrual rates'}}, "
            "{'thought': 'Clarify whether PTO accrual rates vary by seniority.', 'tool_name': 'review_seniority_rules', 'tool_args': {}}, "
            "{'thought': 'Identify carryover rules for unused PTO.', 'tool_name': 'check_carryover_policy', 'tool_args': {'year': 'current year'}}, "
            "{'thought': 'Determine policies on extended leave requests.', 'tool_name': 'search_policy', 'tool_args': {'topic': 'PTO leave request process'}}, "
            "{'thought': 'Check the notification requirements for extended PTO.', 'tool_name': 'notify_manager', 'tool_args': {'type': 'extended leave'}}, "
            "{'thought': 'Investigate overlap between PTO and sick leave.', 'tool_name': 'combine_leave', 'tool_args': {'types': ['PTO', 'sick leave']}}, "
            "{'thought': 'Explore how PTO interacts with parental leave.', 'tool_name': 'combine_leave', 'tool_args': {'types': ['PTO', 'parental leave']}}, "
            "{'thought': 'Fetch the company calendar to determine critical business periods.', 'tool_name': 'fetch_calendar', 'tool_args': {'year': 'current year'}}, "
            "{'thought': 'Verify restrictions on PTO usage during holidays.', 'tool_name': 'search_policy', 'tool_args': {'topic': 'holiday restrictions on PTO'}}, "
            "{'thought': 'Confirm whether unused PTO is compensated upon termination.', 'tool_name': 'verify_compensation', 'tool_args': {'scenario': 'termination'}}, "
            "{'thought': 'Check if PTO is compensated differently upon resignation.', 'tool_name': 'verify_compensation', 'tool_args': {'scenario': 'resignation'}}, "
            "{'thought': 'Review if accrual caps limit PTO earnings.', 'tool_name': 'calculate_accrual', 'tool_args': {'cap': True}}, "
            "{'thought': 'Investigate whether senior employees receive additional PTO benefits.', 'tool_name': 'review_seniority_rules', 'tool_args': {'seniority_level': 'high'}}, "
            "{'thought': 'Assess policy transparency in PTO documentation.', 'tool_name': 'search_policy', 'tool_args': {'topic': 'PTO documentation clarity'}}, "
            "{'thought': 'Explore how leave types can be optimized together.', 'tool_name': 'combine_leave', 'tool_args': {'types': ['PTO', 'other leave']}}, "
            "{'thought': 'Check historical trends in PTO policy changes.', 'tool_name': 'get_docs', 'tool_args': {'document': 'PTO history'}}"
            "]"
        ),
    )
    assert prediction.next_tool_name in next_tool_names
