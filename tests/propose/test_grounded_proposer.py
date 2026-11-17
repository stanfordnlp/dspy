import pytest

import dspy
from dspy.adapters.types.base_type import CUSTOM_TYPE_START_IDENTIFIER
from dspy.adapters.types.document import Document
from dspy.predict import Predict
from dspy.propose.dataset_summary_generator import sanitize_dataset_examples_repr
from dspy.propose.grounded_proposer import GroundedProposer
from dspy.propose.utils import create_example_string, format_proposer_field_value
from dspy.utils.dummies import DummyLM


@pytest.mark.parametrize(
    "demo_candidates",
    [
        None,
        [[[dspy.Example(question="What is the capital of France?", answer="Paris")]]],
    ],
)
def test_propose_instructions_for_program(demo_candidates):
    # Set large number here so that lm always returns the same response
    prompt_model = DummyLM([{"proposed_instruction": "instruction"}] * 10)
    program = Predict("question -> answer")
    trainset = []

    proposer = GroundedProposer(prompt_model=prompt_model, program=program, trainset=trainset, verbose=False)
    result = proposer.propose_instructions_for_program(
        trainset=trainset, program=program, demo_candidates=demo_candidates, trial_logs={}, N=1
    )
    assert isinstance(result, dict)
    assert len(result) == len(program.predictors())
    for pred_instructions in result.values():
        assert pred_instructions == ["instruction"]


@pytest.mark.parametrize(
    "demo_candidates",
    [
        None,
        [[[dspy.Example(question="What is the capital of France?", answer="Paris")]]],
    ],
)
def test_propose_instruction_for_predictor(demo_candidates):
    class TrackingDummyLM(DummyLM):
        def copy(self, **kwargs):
            self.last_copy_kwargs = kwargs
            return super().copy(**kwargs)

    prompt_model = TrackingDummyLM([{"proposed_instruction": "instruction"}] * 10)
    program = Predict("question -> answer")

    proposer = GroundedProposer(
        prompt_model=prompt_model,
        program=program,
        trainset=[],
        verbose=False,
        init_temperature=0.7,
    )
    result = proposer.propose_instruction_for_predictor(
        program=program,
        predictor=None,
        pred_i=0,
        demo_candidates=demo_candidates,
        demo_set_i=0,
        trial_logs={},
        tip=None,
    )
    assert result == "instruction"
    assert prompt_model.last_copy_kwargs["temperature"] == 0.7


def test_format_proposer_field_value_with_image():
    """format_proposer_field_value must not leak base64 or markers into prompts."""
    img = dspy.Image("data:image/png;base64," + "A" * 10000)
    formatted = format_proposer_field_value(img)
    assert len(formatted) < 100
    assert "IMAGE_BASE_64_ENCODED(10000)" in formatted
    assert CUSTOM_TYPE_START_IDENTIFIER not in formatted
    assert "A" * 100 not in formatted


def test_format_proposer_field_value_with_document():
    """Document fields must use compact string representation, not full contents."""
    doc = Document(data="Lorem ipsum dolor sit amet " * 500, title="Astronomy Guide")
    formatted = format_proposer_field_value(doc)
    assert len(formatted) < 100
    assert "Astronomy Guide" in formatted
    assert "chars" in formatted
    assert "Lorem ipsum" not in formatted
    assert CUSTOM_TYPE_START_IDENTIFIER not in formatted


def test_format_proposer_field_value_containers_preserve_quotes():
    """Containers must preserve string quotes and distinguish types."""
    assert format_proposer_field_value(["a", "b"]) == "['a', 'b']"
    assert format_proposer_field_value(("a", "b")) == "('a', 'b')"
    assert format_proposer_field_value(["1"]) == "['1']"
    assert format_proposer_field_value([1]) == "[1]"
    assert format_proposer_field_value(["1"]) != format_proposer_field_value([1])
    assert format_proposer_field_value({"k": "v"}) == "{'k': 'v'}"

    img = dspy.Image("data:image/png;base64," + "A" * 1000)
    in_list = format_proposer_field_value([img])
    assert "IMAGE_BASE_64_ENCODED(1000)" in in_list
    assert CUSTOM_TYPE_START_IDENTIFIER not in in_list


def test_create_example_string_multimodal():
    """create_example_string must sanitize multimodal fields to avoid bloat."""
    class VisualQA(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    example = dspy.Example(
        image=dspy.Image("data:image/png;base64," + "B" * 20000),
        question="What color is the sky?",
        answer="Blue",
    )
    result = create_example_string(VisualQA.fields, example)
    assert len(result) < 250
    assert CUSTOM_TYPE_START_IDENTIFIER not in result
    assert "IMAGE_BASE_64_ENCODED(20000)" in result
    assert "B" * 100 not in result


def test_propose_instructions_multimodal_no_context_blowup():
    """GroundedProposer must propose instructions without image part expansion."""
    class VisualQA(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    program = Predict(VisualQA)
    prompt_model = DummyLM(
        [{"proposed_instruction": "Answer questions about images accurately."}] * 20
    )

    demo = dspy.Example(
        image=dspy.Image("data:image/png;base64," + "C" * 15000),
        question="Identify the animal in the image.",
        answer="Elephant",
        augmented=True,
    ).with_inputs("image", "question")

    # Provide two candidate sets so demo_set_i=1 is exercised
    # (set 0 defaults to no demos).
    demo_candidates = [[[demo], [demo]]]
    proposer = GroundedProposer(
        prompt_model=prompt_model,
        program=program,
        trainset=[],
        verbose=False,
    )
    result = proposer.propose_instructions_for_program(
        trainset=[],
        program=program,
        demo_candidates=demo_candidates,
        trial_logs={},
        N=2,
    )
    assert isinstance(result, dict)
    assert len(result[0]) == 2

    found_compact_image_demo = False
    for call in prompt_model.history:
        messages = call.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                if "IMAGE_BASE_64_ENCODED(15000)" in content:
                    found_compact_image_demo = True
                assert CUSTOM_TYPE_START_IDENTIFIER not in content
                assert "C" * 100 not in content

        if "request" in call and hasattr(call["request"], "messages"):
            for message in call["request"].messages:
                for part in getattr(message, "parts", []):
                    assert getattr(part, "type", "text") == "text"

    assert found_compact_image_demo, "Compact image demo must reach prompt model"


def test_sanitize_dataset_examples_repr():
    """sanitize_dataset_examples_repr replaces markers with compact text."""
    raw = (
        f"{CUSTOM_TYPE_START_IDENTIFIER}[{{\"type\": \"image\"}}]"
        f"{CUSTOM_TYPE_START_IDENTIFIER} data:image/png;base64,{'D' * 500}"
    )
    cleaned = sanitize_dataset_examples_repr(raw)
    assert CUSTOM_TYPE_START_IDENTIFIER not in cleaned
    assert "[CUSTOM-TYPE-START]" in cleaned
    assert "data:<BASE64_DATA_OMITTED>" in cleaned
    assert "D" * 100 not in cleaned

