import copy
import enum
from datetime import datetime

import pydantic
import pytest
import ujson

import dspy
from dspy import Predict, Signature
from dspy.utils.dummies import DummyLM


def test_initialization_with_string_signature():
    signature_string = "input1, input2 -> output"
    predict = Predict(signature_string)
    expected_instruction = "Given the fields `input1`, `input2`, produce the fields `output`."
    assert predict.signature.instructions == expected_instruction
    assert predict.signature.instructions == Signature(signature_string).instructions


def test_reset_method():
    predict_instance = Predict("input -> output")
    predict_instance.lm = "modified"
    predict_instance.traces = ["trace"]
    predict_instance.train = ["train"]
    predict_instance.demos = ["demo"]
    predict_instance.reset()
    assert predict_instance.lm is None
    assert predict_instance.traces == []
    assert predict_instance.train == []
    assert predict_instance.demos == []


def test_lm_after_dump_and_load_state():
    predict_instance = Predict("input -> output")
    predict_instance.lm = "lm_state"
    dumped_state = predict_instance.dump_state()
    new_instance = Predict("input -> output")
    new_instance.load_state(dumped_state)
    assert new_instance.lm == "lm_state"


def test_call_method():
    predict_instance = Predict("input -> output")
    lm = DummyLM([{"output": "test output"}])
    dspy.settings.configure(lm=lm)
    result = predict_instance(input="test input")
    assert result.output == "test output"


def test_instructions_after_dump_and_load_state():
    predict_instance = Predict(Signature("input -> output", "original instructions"))
    dumped_state = predict_instance.dump_state()
    new_instance = Predict(Signature("input -> output", "new instructions"))
    new_instance.load_state(dumped_state)
    assert new_instance.signature.instructions == "original instructions"


def test_demos_after_dump_and_load_state():
    class TranslateToEnglish(dspy.Signature):
        """Translate content from a language to English."""

        content: str = dspy.InputField()
        language: str = dspy.InputField()
        translation: str = dspy.OutputField()

    original_instance = Predict(TranslateToEnglish)
    original_instance.demos = [
        dspy.Example(
            content="¿Qué tal?",
            language="SPANISH",
            translation="Hello there",
        ).with_inputs("content", "language"),
    ]

    dumped_state = original_instance.dump_state()
    assert len(dumped_state["demos"]) == len(original_instance.demos)
    assert dumped_state["demos"][0]["content"] == original_instance.demos[0].content

    saved_state = ujson.dumps(dumped_state)
    loaded_state = ujson.loads(saved_state)

    new_instance = Predict(TranslateToEnglish)
    new_instance.load_state(loaded_state)
    assert len(new_instance.demos) == len(original_instance.demos)
    # Demos don't need to keep the same types after saving and loading the state.
    assert new_instance.demos[0]["content"] == original_instance.demos[0].content


# def test_typed_demos_after_dump_and_load_state():
#     class TypedTranslateToEnglish(dspy.Signature):
#         """Translate content from a language to English."""

#         class Input(pydantic.BaseModel):
#             content: str
#             language: str

#         class Output(pydantic.BaseModel):
#             translation: str

#         input: Input = dspy.InputField()
#         output: Output = dspy.OutputField()

#     original_instance = TypedPredictor(TypedTranslateToEnglish).predictor
#     original_instance.demos = [
#         dspy.Example(
#             input=TypedTranslateToEnglish.Input(
#                 content="¿Qué tal?",
#                 language="SPANISH",
#             ),
#             output=TypedTranslateToEnglish.Output(
#                 translation="Hello there",
#             ),
#         ).with_inputs("input"),
#     ]

#     dumped_state = original_instance.dump_state()
#     assert len(dumped_state["demos"]) == len(original_instance.demos)
#     assert dumped_state["demos"][0]["input"] == original_instance.demos[0].input.model_dump_json()

#     saved_state = ujson.dumps(dumped_state)
#     loaded_state = ujson.loads(saved_state)

#     new_instance = TypedPredictor(TypedTranslateToEnglish).predictor
#     new_instance.load_state(loaded_state)
#     assert len(new_instance.demos) == len(original_instance.demos)
#     # Demos don't need to keep the same types after saving and loading the state.
#     assert new_instance.demos[0]["input"] == original_instance.demos[0].input.model_dump_json()


def test_signature_fields_after_dump_and_load_state(tmp_path):
    class CustomSignature(dspy.Signature):
        """I am just an instruction."""

        sentence = dspy.InputField(desc="I am an innocent input!")
        sentiment = dspy.OutputField()

    file_path = tmp_path / "tmp.json"
    original_instance = Predict(CustomSignature)
    original_instance.save(file_path)

    class CustomSignature2(dspy.Signature):
        """I am not a pure instruction."""

        sentence = dspy.InputField(desc="I am a malicious input!")
        sentiment = dspy.OutputField(desc="I am a malicious output!", prefix="I am a prefix!")

    new_instance = Predict(CustomSignature2)
    assert new_instance.signature.dump_state() != original_instance.signature.dump_state()
    # After loading, the fields should be the same.
    new_instance.load(file_path)
    assert new_instance.signature.dump_state() == original_instance.signature.dump_state()


def test_forward_method():
    program = Predict("question -> answer")
    dspy.settings.configure(lm=DummyLM([{"answer": "No more responses"}]))
    result = program(question="What is 1+1?").answer
    assert result == "No more responses"


def test_forward_method2():
    program = Predict("question -> answer1, answer2")
    dspy.settings.configure(lm=DummyLM([{"answer1": "my first answer", "answer2": "my second answer"}]))
    result = program(question="What is 1+1?")
    assert result.answer1 == "my first answer"
    assert result.answer2 == "my second answer"


def test_config_management():
    predict_instance = Predict("input -> output")
    predict_instance.update_config(new_key="value")
    config = predict_instance.get_config()
    assert "new_key" in config and config["new_key"] == "value"


def test_multi_output():
    program = Predict("question -> answer", n=2)
    dspy.settings.configure(lm=DummyLM([{"answer": "my first answer"}, {"answer": "my second answer"}]))
    results = program(question="What is 1+1?")
    assert results.completions.answer[0] == "my first answer"
    assert results.completions.answer[1] == "my second answer"


def test_multi_output2():
    program = Predict("question -> answer1, answer2", n=2)
    dspy.settings.configure(
        lm=DummyLM(
            [
                {"answer1": "my 0 answer", "answer2": "my 2 answer"},
                {"answer1": "my 1 answer", "answer2": "my 3 answer"},
            ],
        )
    )
    results = program(question="What is 1+1?")
    assert results.completions.answer1[0] == "my 0 answer"
    assert results.completions.answer1[1] == "my 1 answer"
    assert results.completions.answer2[0] == "my 2 answer"
    assert results.completions.answer2[1] == "my 3 answer"


def test_datetime_inputs_and_outputs():
    # Define a model for datetime inputs and outputs
    class TimedEvent(pydantic.BaseModel):
        event_name: str
        event_time: datetime

    class TimedSignature(dspy.Signature):
        events: list[TimedEvent] = dspy.InputField()
        summary: str = dspy.OutputField()
        next_event_time: datetime = dspy.OutputField()

    program = Predict(TimedSignature)

    lm = DummyLM(
        [
            {
                "reasoning": "Processed datetime inputs",
                "summary": "All events are processed",
                "next_event_time": "2024-11-27T14:00:00",
            }
        ]
    )
    dspy.settings.configure(lm=lm)

    output = program(
        events=[
            TimedEvent(event_name="Event 1", event_time=datetime(2024, 11, 25, 10, 0, 0)),
            TimedEvent(event_name="Event 2", event_time=datetime(2024, 11, 25, 15, 30, 0)),
        ]
    )
    assert output.summary == "All events are processed"
    assert output.next_event_time == datetime(2024, 11, 27, 14, 0, 0)


def test_explicitly_valued_enum_inputs_and_outputs():
    class Status(enum.Enum):
        PENDING = "pending"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"

    class StatusSignature(dspy.Signature):
        current_status: Status = dspy.InputField()
        next_status: Status = dspy.OutputField()

    program = Predict(StatusSignature)

    lm = DummyLM(
        [
            {
                "reasoning": "The current status is 'PENDING', advancing to 'IN_PROGRESS'.",
                "next_status": "in_progress",
            }
        ]
    )
    dspy.settings.configure(lm=lm)

    output = program(current_status=Status.PENDING)
    assert output.next_status == Status.IN_PROGRESS


def test_enum_inputs_and_outputs_with_shared_names_and_values():
    class TicketStatus(enum.Enum):
        OPEN = "CLOSED"
        CLOSED = "RESOLVED"
        RESOLVED = "OPEN"

    class TicketStatusSignature(dspy.Signature):
        current_status: TicketStatus = dspy.InputField()
        next_status: TicketStatus = dspy.OutputField()

    program = Predict(TicketStatusSignature)

    # Mock reasoning and output
    lm = DummyLM(
        [
            {
                "reasoning": "The ticket is currently 'OPEN', transitioning to 'CLOSED'.",
                "next_status": "RESOLVED",  # Refers to TicketStatus.CLOSED by value
            }
        ]
    )
    dspy.settings.configure(lm=lm)

    output = program(current_status=TicketStatus.OPEN)
    assert output.next_status == TicketStatus.CLOSED  # By value


def test_auto_valued_enum_inputs_and_outputs():
    Status = enum.Enum("Status", ["PENDING", "IN_PROGRESS", "COMPLETED"])

    class StatusSignature(dspy.Signature):
        current_status: Status = dspy.InputField()
        next_status: Status = dspy.OutputField()

    program = Predict(StatusSignature)

    lm = DummyLM(
        [
            {
                "reasoning": "The current status is 'PENDING', advancing to 'IN_PROGRESS'.",
                "next_status": "IN_PROGRESS",  # Use the auto-assigned value for IN_PROGRESS
            }
        ]
    )
    dspy.settings.configure(lm=lm)

    output = program(current_status=Status.PENDING)
    assert output.next_status == Status.IN_PROGRESS


def test_named_predictors():
    class MyModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.inner = Predict("question -> answer")

    program = MyModule()
    assert program.named_predictors() == [("inner", program.inner)]

    # Check that it also works the second time.
    program2 = copy.deepcopy(program)
    assert program2.named_predictors() == [("inner", program2.inner)]


def test_output_only():
    class OutputOnlySignature(dspy.Signature):
        output = dspy.OutputField()

    predictor = Predict(OutputOnlySignature)

    lm = DummyLM([{"output": "short answer"}])
    dspy.settings.configure(lm=lm)
    assert predictor().output == "short answer"


def test_load_state_chaining():
    """Test that load_state returns self for chaining."""
    original = Predict("question -> answer")
    original.demos = [{"question": "test", "answer": "response"}]
    state = original.dump_state()

    new_instance = Predict("question -> answer").load_state(state)
    assert new_instance is not None
    assert new_instance.demos == original.demos
