import copy
import enum
from datetime import datetime

import pydantic
import pytest
import ujson
import os

import dspy
from dspy import Predict, Signature
from dspy.utils.dummies import DummyLM
from unittest.mock import patch, MagicMock, Mock


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
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        model_type="chat",
        temperature=1,
        max_tokens=100,
        num_retries=10,
    )
    predict_instance.lm = lm
    expected_lm_state = {
        "model": "openai/gpt-4o-mini",
        "model_type": "chat",
        "temperature": 1,
        "max_tokens": 100,
        "num_retries": 10,
        "cache": True,
        "cache_in_memory": True,
        "finetuning_model": None,
        "launch_kwargs": {},
        "train_kwargs": {},
    }
    assert lm.dump_state() == expected_lm_state
    dumped_state = predict_instance.dump_state()
    new_instance = Predict("input -> output")
    new_instance.load_state(dumped_state)
    assert new_instance.lm.dump_state() == expected_lm_state


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


def test_typed_demos_after_dump_and_load_state():
    class Item(pydantic.BaseModel):
        name: str
        quantity: int

    class InventorySignature(dspy.Signature):
        """Handle inventory items and their translations."""

        items: list[Item] = dspy.InputField()
        language: str = dspy.InputField()
        translated_items: list[Item] = dspy.OutputField()
        total_quantity: int = dspy.OutputField()

    original_instance = Predict(InventorySignature)
    original_instance.demos = [
        dspy.Example(
            items=[Item(name="apple", quantity=5), Item(name="banana", quantity=3)],
            language="SPANISH",
            translated_items=[Item(name="manzana", quantity=5), Item(name="plátano", quantity=3)],
            total_quantity=8,
        ).with_inputs("items", "language"),
    ]

    # Test dump_state
    dumped_state = original_instance.dump_state()
    assert len(dumped_state["demos"]) == len(original_instance.demos)
    # Verify the input items were properly serialized
    assert isinstance(dumped_state["demos"][0]["items"], list)
    assert len(dumped_state["demos"][0]["items"]) == 2
    assert dumped_state["demos"][0]["items"][0] == {"name": "apple", "quantity": 5}

    # Test serialization/deserialization
    saved_state = ujson.dumps(dumped_state)
    loaded_state = ujson.loads(saved_state)

    # Test load_state
    new_instance = Predict(InventorySignature)
    new_instance.load_state(loaded_state)
    assert len(new_instance.demos) == len(original_instance.demos)

    # Verify the structure is maintained after loading
    loaded_demo = new_instance.demos[0]
    assert isinstance(loaded_demo["items"], list)
    assert len(loaded_demo["items"]) == 2
    assert loaded_demo["items"][0]["name"] == "apple"
    assert loaded_demo["items"][0]["quantity"] == 5
    assert loaded_demo["items"][1]["name"] == "banana"
    assert loaded_demo["items"][1]["quantity"] == 3

    # Verify output items were also properly maintained
    assert isinstance(loaded_demo["translated_items"], list)
    assert len(loaded_demo["translated_items"]) == 2
    assert loaded_demo["translated_items"][0]["name"] == "manzana"
    assert loaded_demo["translated_items"][1]["name"] == "plátano"


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


@pytest.mark.parametrize("filename", ["model.json", "model.pkl"])
def test_lm_field_after_dump_and_load_state(tmp_path, filename):
    file_path = tmp_path / filename
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        model_type="chat",
        temperature=1,
        max_tokens=100,
        num_retries=10,
    )
    original_predict = dspy.Predict("q->a")
    original_predict.lm = lm

    original_predict.save(file_path)

    assert file_path.exists()

    loaded_predict = dspy.Predict("q->a")
    loaded_predict.load(file_path)

    assert original_predict.dump_state() == loaded_predict.dump_state()


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


@pytest.mark.parametrize("adapter_type", ["chat", "json"])
def test_call_predict_with_chat_history(adapter_type):
    class SpyLM(dspy.LM):
        def __init__(self, *args, return_json=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = []
            self.return_json = return_json

        def __call__(self, prompt=None, messages=None, **kwargs):
            self.calls.append({"prompt": prompt, "messages": messages, "kwargs": kwargs})
            if self.return_json:
                return ["{'answer':'100%'}"]
            return ["[[ ## answer ## ]]\n100%!"]

    class MySignature(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()

    program = Predict(MySignature)

    if adapter_type == "chat":
        lm = SpyLM("dummy_model")
        dspy.settings.configure(adapter=dspy.ChatAdapter(), lm=lm)
    else:
        lm = SpyLM("dummy_model", return_json=True)
        dspy.settings.configure(adapter=dspy.JSONAdapter(), lm=lm)

    program(
        question="are you sure that's correct?",
        history=dspy.History(messages=[{"question": "what's the capital of france?", "answer": "paris"}]),
    )

    # Verify the LM was called with correct messages
    assert len(lm.calls) == 1
    messages = lm.calls[0]["messages"]

    assert len(messages) == 4

    assert "what's the capital of france?" in messages[1]["content"]
    assert "paris" in messages[2]["content"]
    assert "are you sure that's correct" in messages[3]["content"]


@pytest.mark.parametrize("adapter_type", ["chat", "json"])
def test_field_constraints(adapter_type):
    class SpyLM(dspy.LM):
        def __init__(self, *args, return_json=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = []
            self.return_json = return_json

        def __call__(self, prompt=None, messages=None, **kwargs):
            self.calls.append({"prompt": prompt, "messages": messages, "kwargs": kwargs})
            if self.return_json:
                return ["{'score':'0.5', 'count':'2'}"]
            return ["[[ ## score ## ]]\n0.5\n[[ ## count ## ]]\n2"]

    class ConstrainedSignature(dspy.Signature):
        """Test signature with constrained fields."""

        # Input with length and value constraints
        text: str = dspy.InputField(min_length=5, max_length=100, desc="Input text")
        number: int = dspy.InputField(gt=0, lt=10, desc="A number between 0 and 10")

        # Output with multiple constraints
        score: float = dspy.OutputField(ge=0.0, le=1.0, desc="Score between 0 and 1")
        count: int = dspy.OutputField(multiple_of=2, desc="Even number count")

    program = Predict(ConstrainedSignature)
    lm = SpyLM("dummy_model")
    if adapter_type == "chat":
        lm = SpyLM("dummy_model")
        dspy.settings.configure(adapter=dspy.ChatAdapter(), lm=lm)
    else:
        lm = SpyLM("dummy_model", return_json=True)
        dspy.settings.configure(adapter=dspy.JSONAdapter(), lm=lm)

    # Call the predictor to trigger instruction generation
    program(text="hello world", number=5)

    # Get the system message containing the instructions
    system_message = lm.calls[0]["messages"][0]["content"]

    # Verify constraints are included in the field descriptions
    assert "minimum length: 5" in system_message
    assert "maximum length: 100" in system_message
    assert "greater than: 0" in system_message
    assert "less than: 10" in system_message
    assert "greater than or equal to: 0.0" in system_message
    assert "less than or equal to: 1.0" in system_message
    assert "a multiple of the given number: 2" in system_message


@pytest.mark.skipif(os.environ.get("OPENAI_API_KEY") is None, reason="Skipping if OPENAI_API_KEY is not set")
def test_litellm_cache_initialization_failure():
    """Test that DSPy handles litellm cache initialization failure gracefully."""
    # Mock Cache to raise a permission error
    mock_cache = MagicMock()
    mock_cache.side_effect = PermissionError("Permission denied")

    import litellm

    # Normal import should have set litellm.cache
    assert litellm.cache is not None

    with patch("litellm.caching.Cache", mock_cache):
        import importlib
        import dspy.clients

        importlib.reload(dspy.clients)

    # On cache initialization failure, litellm.cache should be set to None
    assert litellm.cache is None

    # Create a simple predictor to verify it works without cache
    predictor = Predict("question -> answer")
    dspy.settings.configure(lm=dspy.LM(model="openai/gpt-4o-mini"))

    # No exception should be raised when litellm.cache is None even if we try to use the cache.
    assert dspy.settings.lm.cache == True
    predictor(question="test")
