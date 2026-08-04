import logging
import re
from typing import Optional

import pytest
from pydantic import BaseModel, model_validator

import dspy
import dspy.adapters.base as adapter_base
import dspy.adapters.utils as adapter_utils
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import ContextWindowExceededError


@pytest.mark.extra
def test_tool_observation_preserves_custom_type():
    pytest.importorskip("PIL.Image")
    from PIL import Image

    captured_calls = []

    class SpyChatAdapter(dspy.ChatAdapter):
        def format_user_message_content(self, signature, inputs, *args, **kwargs):
            captured_calls.append((signature, dict(inputs)))
            return super().format_user_message_content(signature, inputs, *args, **kwargs)

    def make_images():
        return dspy.Image("https://example.com/test.png"), dspy.Image(Image.new("RGB", (100, 100), "red"))

    adapter = SpyChatAdapter()
    lm = DummyLM(
        [
            {
                "next_thought": "I should call the image tool.",
                "next_tool_name": "make_images",
                "next_tool_args": {},
            },
            {
                "next_thought": "I now have the image so I can finish.",
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            {"reasoning": "image ready", "answer": "done"},
        ],
        adapter=adapter,
    )
    dspy.configure(lm=lm, adapter=adapter)

    react = dspy.ReAct("question -> answer", tools=[make_images])
    react(question="Draw me something red")

    sigs_with_obs = [sig for sig, inputs in captured_calls if "observation_0" in inputs]
    assert sigs_with_obs, "Expected ReAct to format a trajectory containing observation_0"

    observation_content = lm.history[1]["messages"][1]["content"]
    assert sum(1 for part in observation_content if isinstance(part, dict) and part.get("type") == "image_url") == 2


def test_tool_calling_with_pydantic_args():
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: dict[str, str]

    def write_invitation_letter(participant_name: str, event_info: CalendarEvent):
        if participant_name not in event_info.participants:
            return None
        return f"It's my honor to invite {participant_name} to event {event_info.name} on {event_info.date}"

    class InvitationSignature(dspy.Signature):
        participant_name: str = dspy.InputField(desc="The name of the participant to invite")
        event_info: CalendarEvent = dspy.InputField(desc="The information about the event")
        invitation_letter: str = dspy.OutputField(desc="The invitation letter to be sent to the participant")

    react = dspy.ReAct(InvitationSignature, tools=[write_invitation_letter])

    lm = DummyLM(
        [
            {
                "next_thought": "I need to write an invitation letter for Alice to the Science Fair event.",
                "next_tool_name": "write_invitation_letter",
                "next_tool_args": {
                    "participant_name": "Alice",
                    "event_info": {
                        "name": "Science Fair",
                        "date": "Friday",
                        "participants": {"Alice": "female", "Bob": "male"},
                    },
                },
            },
            {
                "next_thought": (
                    "I have successfully written the invitation letter for Alice to the Science Fair. Now "
                    "I can finish the task."
                ),
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            {
                "reasoning": "This is a very rigorous reasoning process, trust me bro!",
                "invitation_letter": "It's my honor to invite Alice to the Science Fair event on Friday.",
            },
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(
        participant_name="Alice",
        event_info=CalendarEvent(
            name="Science Fair",
            date="Friday",
            participants={"Alice": "female", "Bob": "male"},
        ),
    )
    assert outputs.invitation_letter == "It's my honor to invite Alice to the Science Fair event on Friday."

    expected_trajectory = {
        "thought_0": "I need to write an invitation letter for Alice to the Science Fair event.",
        "tool_name_0": "write_invitation_letter",
        "tool_args_0": {
            "participant_name": "Alice",
            "event_info": {
                "name": "Science Fair",
                "date": "Friday",
                "participants": {"Alice": "female", "Bob": "male"},
            },
        },
        "observation_0": "It's my honor to invite Alice to event Science Fair on Friday",
        "thought_1": "I have successfully written the invitation letter for Alice to the Science Fair. Now I can finish the task.",
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": "Completed.",
    }
    assert outputs.trajectory == expected_trajectory


def test_react_with_tools_skips_native_response_issubclass_for_generic_alias(monkeypatch):
    def get_user_info(name: str):
        return {"name": name}

    class CustomerService(dspy.Signature):
        user_request: str = dspy.InputField()
        process_result: str = dspy.OutputField()

    react = dspy.ReAct(CustomerService, tools=[get_user_info])
    problem_annotation = react.react.signature.output_fields["next_tool_args"].annotation

    def guarded_issubclass(cls, class_or_tuple):
        if cls == problem_annotation:
            raise TypeError("issubclass() arg 1 must be a class")
        return issubclass(cls, class_or_tuple)

    monkeypatch.setattr(adapter_base, "issubclass", guarded_issubclass, raising=False)
    monkeypatch.setattr(adapter_utils, "issubclass", guarded_issubclass, raising=False)

    lm = DummyLM(
        [
            {
                "next_thought": "I should look up the user first.",
                "next_tool_name": "get_user_info",
                "next_tool_args": {"name": "Adam"},
            },
            {
                "next_thought": "I have the information I need, so I can finish now.",
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            {
                "reasoning": "I fetched the user profile and can answer the request.",
                "process_result": "Resolved Adam's request.",
            },
        ]
    )

    with dspy.context(lm=lm):
        result = react(user_request="Help me, my name is Adam")

    assert result.process_result == "Resolved Adam's request."
    assert result.trajectory["tool_name_0"] == "get_user_info"
    assert result.trajectory["tool_args_0"] == {"name": "Adam"}


def test_tool_calling_without_typehint():
    def foo(a, b):
        """Add two numbers."""
        return a + b

    react = dspy.ReAct("a, b -> c:int", tools=[foo])
    lm = DummyLM(
        [
            {"next_thought": "I need to add two numbers.", "next_tool_name": "foo", "next_tool_args": {"a": 1, "b": 2}},
            {"next_thought": "I have the sum, now I can finish.", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "I added the numbers successfully", "c": 3},
        ]
    )
    dspy.configure(lm=lm)
    outputs = react(a=1, b=2)

    expected_trajectory = {
        "thought_0": "I need to add two numbers.",
        "tool_name_0": "foo",
        "tool_args_0": {
            "a": 1,
            "b": 2,
        },
        "observation_0": 3,
        "thought_1": "I have the sum, now I can finish.",
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": "Completed.",
    }
    assert outputs.trajectory == expected_trajectory


def test_trajectory_truncation():
    # Create a simple tool for testing
    def echo(text: str) -> str:
        return f"Echoed: {text}"

    # Create ReAct instance with our echo tool
    react = dspy.ReAct("input_text -> output_text", tools=[echo])

    # Mock react.react to simulate multiple tool calls
    call_count = 0

    def mock_react(**kwargs):
        nonlocal call_count
        call_count += 1

        if call_count < 3:
            # First 2 calls use the echo tool
            return dspy.Prediction(
                next_thought=f"Thought {call_count}",
                next_tool_name="echo",
                next_tool_args={"text": f"Text {call_count}"},
            )
        elif call_count == 3:
            # The 3rd call raises context window exceeded error
            raise ContextWindowExceededError()
        else:
            # The 4th call finishes
            return dspy.Prediction(next_thought="Final thought", next_tool_name="finish", next_tool_args={})

    react.react = mock_react
    react.extract = lambda **kwargs: dspy.Prediction(output_text="Final output")

    # Call forward and get the result
    result = react(input_text="test input")

    # Verify that older entries in the trajectory were truncated
    assert "thought_0" not in result.trajectory
    assert "thought_2" in result.trajectory
    assert result.output_text == "Final output"


def test_truncate_trajectory_raises_on_single_tool_call():
    # A trajectory with exactly one tool call has 4 keys (thought, tool_name, tool_args,
    # observation). Per truncate_trajectory's own docstring/error message, this is the
    # smallest trajectory that cannot be truncated further, so it must raise instead of
    # silently popping every key and returning an empty trajectory.
    react = dspy.ReAct("input_text -> output_text", tools=[])
    trajectory = {
        "thought_0": "Thought 0",
        "tool_name_0": "finish",
        "tool_args_0": {},
        "observation_0": "Completed.",
    }

    with pytest.raises(ContextWindowExceededError):
        react.truncate_trajectory(trajectory)


def test_truncation_exhausted_raises_context_window_exceeded_error():
    def echo(text: str) -> str:
        return f"Echoed: {text}"

    react = dspy.ReAct("input_text -> output_text", tools=[echo])

    def always_exceed(**kwargs):
        raise ContextWindowExceededError()

    trajectory = {}
    for i in range(4):
        trajectory[f"thought_{i}"] = f"Thought {i}"
        trajectory[f"tool_name_{i}"] = "echo"
        trajectory[f"tool_args_{i}"] = {"text": f"Text {i}"}
        trajectory[f"observation_{i}"] = f"Echoed: Text {i}"

    with pytest.raises(ContextWindowExceededError, match="even after 3 attempts") as exc_info:
        react._call_with_potential_trajectory_truncation(always_exceed, trajectory, input_text="test input")

    assert isinstance(exc_info.value.__cause__, ContextWindowExceededError)


@pytest.mark.asyncio
async def test_context_window_exceeded_after_retries():
    def echo(text: str) -> str:
        return f"Echoed: {text}"

    react = dspy.ReAct("input_text -> output_text", tools=[echo])

    def mock_react(**kwargs):
        raise ContextWindowExceededError()

    # Test sync version
    extract_calls = []

    def mock_extract(**kwargs):
        extract_calls.append(kwargs)
        return dspy.Prediction(output_text="Fallback output")

    react.react = mock_react
    react.extract = mock_extract

    result = react(input_text="test input")
    assert result.trajectory == {}
    assert result.output_text == "Fallback output"
    assert len(extract_calls) == 1
    assert extract_calls[0]["input_text"] == "test input"
    assert "trajectory" in extract_calls[0]

    # Test async version
    async_extract_calls = []

    async def mock_react_async(**kwargs):
        raise ContextWindowExceededError()

    async def mock_extract_async(**kwargs):
        async_extract_calls.append(kwargs)
        return dspy.Prediction(output_text="Fallback output")

    react.react.acall = mock_react_async
    react.extract.acall = mock_extract_async

    result = await react.acall(input_text="test input")
    assert result.trajectory == {}
    assert result.output_text == "Fallback output"
    assert len(async_extract_calls) == 1
    assert async_extract_calls[0]["input_text"] == "test input"
    assert "trajectory" in async_extract_calls[0]


def test_error_retry():
    # --- a tiny tool that always fails -------------------------------------
    def foo(a, b):
        raise Exception("tool error")

    # --- program under test -------------------------------------------------
    react = dspy.ReAct("a, b -> c:int", tools=[foo])
    lm = DummyLM(
        [
            {
                "next_thought": "I need to add two numbers.",
                "next_tool_name": "foo",
                "next_tool_args": {"a": 1, "b": 2},
            },
            {
                "next_thought": "I need to add two numbers.",
                "next_tool_name": "foo",
                "next_tool_args": {"a": 1, "b": 2},
            },
            # (The model *would* succeed on the 3rd turn, but max_iters=2 stops earlier.)
            {"reasoning": "I added the numbers successfully", "c": 3},
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(a=1, b=2, max_iters=2)
    traj = outputs.trajectory

    # --- exact-match checks (thoughts + tool calls) -------------------------
    control_expected = {
        "thought_0": "I need to add two numbers.",
        "tool_name_0": "foo",
        "tool_args_0": {"a": 1, "b": 2},
        "thought_1": "I need to add two numbers.",
        "tool_name_1": "foo",
        "tool_args_1": {"a": 1, "b": 2},
    }
    for k, v in control_expected.items():
        assert traj[k] == v, f"{k} mismatch"

    # --- flexible checks for observations ----------------------------------
    # We only care that each observation mentions our error string; we ignore
    # any extra traceback detail or differing prefixes.
    for i in range(2):
        obs = traj[f"observation_{i}"]
        assert re.search(r"\btool error\b", obs), f"unexpected observation_{i!r}: {obs}"


def test_tool_error_observation_format():
    def failing_tool():
        raise ValueError("tool blew up")

    react = dspy.ReAct("question -> answer", tools=[failing_tool])
    lm = DummyLM(
        [
            {
                "next_thought": "I will call the tool.",
                "next_tool_name": "failing_tool",
                "next_tool_args": {},
            },
            {"reasoning": "The tool failed.", "answer": "n/a"},
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(question="What happens?", max_iters=1)
    obs = outputs.trajectory["observation_0"]

    assert obs.startswith("Execution error in failing_tool: \nTraceback (most recent call last):")
    assert obs.endswith("ValueError: tool blew up")


@pytest.mark.asyncio
async def test_async_tool_calling_with_pydantic_args():
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: dict[str, str]

    async def write_invitation_letter(participant_name: str, event_info: CalendarEvent):
        if participant_name not in event_info.participants:
            return None
        return f"It's my honor to invite {participant_name} to event {event_info.name} on {event_info.date}"

    class InvitationSignature(dspy.Signature):
        participant_name: str = dspy.InputField(desc="The name of the participant to invite")
        event_info: CalendarEvent = dspy.InputField(desc="The information about the event")
        invitation_letter: str = dspy.OutputField(desc="The invitation letter to be sent to the participant")

    react = dspy.ReAct(InvitationSignature, tools=[write_invitation_letter])

    lm = DummyLM(
        [
            {
                "next_thought": "I need to write an invitation letter for Alice to the Science Fair event.",
                "next_tool_name": "write_invitation_letter",
                "next_tool_args": {
                    "participant_name": "Alice",
                    "event_info": {
                        "name": "Science Fair",
                        "date": "Friday",
                        "participants": {"Alice": "female", "Bob": "male"},
                    },
                },
            },
            {
                "next_thought": (
                    "I have successfully written the invitation letter for Alice to the Science Fair. Now "
                    "I can finish the task."
                ),
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            {
                "reasoning": "This is a very rigorous reasoning process, trust me bro!",
                "invitation_letter": "It's my honor to invite Alice to the Science Fair event on Friday.",
            },
        ]
    )
    with dspy.context(lm=lm):
        outputs = await react.acall(
            participant_name="Alice",
            event_info=CalendarEvent(
                name="Science Fair",
                date="Friday",
                participants={"Alice": "female", "Bob": "male"},
            ),
        )
    assert outputs.invitation_letter == "It's my honor to invite Alice to the Science Fair event on Friday."

    expected_trajectory = {
        "thought_0": "I need to write an invitation letter for Alice to the Science Fair event.",
        "tool_name_0": "write_invitation_letter",
        "tool_args_0": {
            "participant_name": "Alice",
            "event_info": {
                "name": "Science Fair",
                "date": "Friday",
                "participants": {"Alice": "female", "Bob": "male"},
            },
        },
        "observation_0": "It's my honor to invite Alice to event Science Fair on Friday",
        "thought_1": "I have successfully written the invitation letter for Alice to the Science Fair. Now I can finish the task.",
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": "Completed.",
    }
    assert outputs.trajectory == expected_trajectory


class _CountingExtract:
    """Wraps `react.extract` to count how many times the extract fallback runs."""

    def __init__(self, extract):
        self.extract = extract
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return self.extract(**kwargs)

    async def acall(self, **kwargs):
        self.calls += 1
        return await self.extract.acall(**kwargs)


def test_finish_with_typed_args_skips_extract():
    def add(a: int, b: int) -> int:
        return a + b

    react = dspy.ReAct("question -> answer", tools=[add])
    counting_extract = _CountingExtract(react.extract)
    react.extract = counting_extract

    lm = DummyLM(
        [
            {
                "next_thought": "1 + 2 is 3, so I can finish with the answer directly.",
                "next_tool_name": "finish",
                "next_tool_args": {"answer": "The sum is 3."},
            }
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(question="What is 1 + 2?")

    assert outputs.answer == "The sum is 3."
    assert outputs.reasoning == "1 + 2 is 3, so I can finish with the answer directly."
    assert counting_extract.calls == 0
    assert len(lm.history) == 1
    assert outputs.trajectory == {
        "thought_0": "1 + 2 is 3, so I can finish with the answer directly.",
        "tool_name_0": "finish",
        "tool_args_0": {"answer": "The sum is 3."},
        "observation_0": "Completed.",
    }


def test_finish_args_validate_output_field_types():
    class Event(BaseModel):
        name: str
        year: int

    class TypedSignature(dspy.Signature):
        question: str = dspy.InputField()
        title: str = dspy.OutputField()
        count: int = dspy.OutputField()
        tags: list[str] = dspy.OutputField()
        event: Event = dspy.OutputField()

    react = dspy.ReAct(TypedSignature, tools=[])
    lm = DummyLM(
        [
            {
                "next_thought": "All outputs are ready.",
                "next_tool_name": "finish",
                "next_tool_args": {
                    "title": "Science Fair",
                    "count": 2,
                    "tags": ["science", "fair"],
                    "event": {"name": "Science Fair", "year": 2026},
                },
            }
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(question="Describe the event")

    assert outputs.title == "Science Fair"
    assert outputs.count == 2
    assert isinstance(outputs.count, int)
    assert outputs.tags == ["science", "fair"]
    assert isinstance(outputs.event, Event)
    assert outputs.event == Event(name="Science Fair", year=2026)
    assert len(lm.history) == 1


def test_finish_args_coerced_from_strings():
    class Event(BaseModel):
        name: str
        year: int

    class TypedSignature(dspy.Signature):
        question: str = dspy.InputField()
        count: int = dspy.OutputField()
        event: Event = dspy.OutputField()

    react = dspy.ReAct(TypedSignature, tools=[])
    lm = DummyLM(
        [
            {
                "next_thought": "All outputs are ready.",
                "next_tool_name": "finish",
                "next_tool_args": {
                    "count": "7",
                    "event": '{"name": "Science Fair", "year": 2026}',
                },
            }
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(question="Describe the event")

    assert outputs.count == 7
    assert isinstance(outputs.count, int)
    assert outputs.event == Event(name="Science Fair", year=2026)
    assert len(lm.history) == 1
    assert outputs.trajectory["observation_0"] == "Completed."


def test_finish_with_empty_args_falls_back_to_extract():
    react = dspy.ReAct("question -> answer", tools=[])
    lm = DummyLM(
        [
            {"next_thought": "I know the answer.", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "Extracted reasoning.", "answer": "extracted answer"},
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(question="q")

    assert outputs.answer == "extracted answer"
    assert outputs.reasoning == "Extracted reasoning."
    assert len(lm.history) == 2
    assert outputs.trajectory["observation_0"] == "Completed."


def test_finish_with_partial_args_falls_back_to_extract():
    react = dspy.ReAct("question -> answer, source", tools=[])
    lm = DummyLM(
        [
            {
                "next_thought": "Finishing with only part of the outputs.",
                "next_tool_name": "finish",
                "next_tool_args": {"answer": "partial"},
            },
            {"reasoning": "Extracted.", "answer": "extracted answer", "source": "extracted source"},
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(question="q")

    assert outputs.answer == "extracted answer"
    assert outputs.source == "extracted source"
    assert len(lm.history) == 2


def test_finish_with_uncoercible_args_falls_back_to_extract():
    react = dspy.ReAct("question -> count: int", tools=[])
    lm = DummyLM(
        [
            {
                "next_thought": "Finishing with a bad value.",
                "next_tool_name": "finish",
                "next_tool_args": {"count": "not a number"},
            },
            {"reasoning": "Extracted.", "count": 42},
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(question="q")

    assert outputs.count == 42
    assert len(lm.history) == 2


def test_finish_args_raising_non_validation_error_falls_back_to_extract():
    class Item(BaseModel):
        name: str

        @model_validator(mode="before")
        @classmethod
        def normalize(cls, value):
            if isinstance(value, dict):
                return value
            # Pydantic only converts ValueError/AssertionError into a ValidationError, so this
            # raises AttributeError straight out of `validate_python` for non-string values.
            return {"name": value.strip()}

    class ItemSignature(dspy.Signature):
        question: str = dspy.InputField()
        item: Item = dspy.OutputField()

    react = dspy.ReAct(ItemSignature, tools=[])
    lm = DummyLM(
        [
            {
                "next_thought": "Finishing with a value the validator chokes on.",
                "next_tool_name": "finish",
                "next_tool_args": {"item": 123},
            },
            {"reasoning": "Extracted.", "item": {"name": "extracted"}},
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(question="q")

    assert outputs.item == Item(name="extracted")
    assert len(lm.history) == 2


def test_unrepresentable_output_annotation_advertises_string_schema_and_logs(caplog):
    class Node(BaseModel):
        value: str
        children: list["Node"] = []

    class TreeSignature(dspy.Signature):
        question: str = dspy.InputField()
        tree: Node = dspy.OutputField()

    # The `dspy` logger does not propagate to the root logger that caplog installs itself on.
    dspy_logger = logging.getLogger("dspy")
    dspy_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.DEBUG, logger="dspy.predict.react"):
            react = dspy.ReAct(TreeSignature, tools=[])
    finally:
        dspy_logger.removeHandler(caplog.handler)

    assert react.tools["finish"].args["tree"] == {"type": "string"}
    assert any("Node" in record.message for record in caplog.records)

    lm = DummyLM(
        [
            {
                "next_thought": "The tree is ready.",
                "next_tool_name": "finish",
                "next_tool_args": {"tree": {"value": "root", "children": [{"value": "leaf"}]}},
            }
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(question="q")

    assert outputs.tree == Node(value="root", children=[Node(value="leaf")])
    assert len(lm.history) == 1


def test_finish_with_explicit_none_for_optional_output_skips_extract():
    class OptionalSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: Optional[str] = dspy.OutputField()  # noqa: UP045 - the typing.Optional spelling is what this test covers

    react = dspy.ReAct(OptionalSignature, tools=[])
    lm = DummyLM(
        [
            {"next_thought": "There is no answer.", "next_tool_name": "finish", "next_tool_args": {"answer": None}},
            {"reasoning": "Extracted.", "answer": "EXTRACT_FALLBACK"},
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(question="q")

    assert outputs.answer is None
    assert outputs.reasoning == "There is no answer."
    assert len(lm.history) == 1


def test_finish_with_explicit_none_for_union_none_output_skips_extract():
    class NullableSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str | None = dspy.OutputField()

    react = dspy.ReAct(NullableSignature, tools=[])
    lm = DummyLM(
        [
            {"next_thought": "There is no answer.", "next_tool_name": "finish", "next_tool_args": {"answer": None}},
            {"reasoning": "Extracted.", "answer": "EXTRACT_FALLBACK"},
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(question="q")

    assert outputs.answer is None
    assert len(lm.history) == 1


def test_finish_missing_optional_output_still_falls_back_to_extract():
    class OptionalSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str | None = dspy.OutputField()

    react = dspy.ReAct(OptionalSignature, tools=[])
    lm = DummyLM(
        [
            {"next_thought": "Finishing without the answer.", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "Extracted.", "answer": "extracted answer"},
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(question="q")

    assert outputs.answer == "extracted answer"
    assert len(lm.history) == 2


def test_max_iters_exhausted_without_finish_still_calls_extract():
    def echo(text: str) -> str:
        return f"Echoed: {text}"

    react = dspy.ReAct("question -> answer", tools=[echo])
    lm = DummyLM(
        [
            {"next_thought": "Echo once.", "next_tool_name": "echo", "next_tool_args": {"text": "a"}},
            {"next_thought": "Echo twice.", "next_tool_name": "echo", "next_tool_args": {"text": "b"}},
            {"reasoning": "Extracted.", "answer": "extracted answer"},
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(question="q", max_iters=2)

    assert outputs.answer == "extracted answer"
    assert len(lm.history) == 3
    assert outputs.trajectory["observation_1"] == "Echoed: b"


def test_finish_fast_path_prefers_output_field_named_reasoning():
    react = dspy.ReAct("question -> reasoning", tools=[])
    lm = DummyLM(
        [
            {
                "next_thought": "The internal thought.",
                "next_tool_name": "finish",
                "next_tool_args": {"reasoning": "The final reasoning output."},
            }
        ]
    )
    dspy.configure(lm=lm)

    outputs = react(question="q")

    assert outputs.reasoning == "The final reasoning output."
    assert len(lm.history) == 1


@pytest.mark.asyncio
async def test_async_finish_with_typed_args_skips_extract():
    def add(a: int, b: int) -> int:
        return a + b

    react = dspy.ReAct("question -> answer", tools=[add])
    counting_extract = _CountingExtract(react.extract)
    react.extract = counting_extract

    lm = DummyLM(
        [
            {
                "next_thought": "1 + 2 is 3, so I can finish with the answer directly.",
                "next_tool_name": "finish",
                "next_tool_args": {"answer": "The sum is 3."},
            }
        ]
    )
    with dspy.context(lm=lm):
        outputs = await react.acall(question="What is 1 + 2?")

    assert outputs.answer == "The sum is 3."
    assert outputs.reasoning == "1 + 2 is 3, so I can finish with the answer directly."
    assert counting_extract.calls == 0
    assert len(lm.history) == 1
    assert outputs.trajectory == {
        "thought_0": "1 + 2 is 3, so I can finish with the answer directly.",
        "tool_name_0": "finish",
        "tool_args_0": {"answer": "The sum is 3."},
        "observation_0": "Completed.",
    }


@pytest.mark.asyncio
async def test_async_finish_args_coerced_to_declared_types():
    class Event(BaseModel):
        name: str
        year: int

    class TypedSignature(dspy.Signature):
        question: str = dspy.InputField()
        count: int = dspy.OutputField()
        event: Event = dspy.OutputField()

    react = dspy.ReAct(TypedSignature, tools=[])
    lm = DummyLM(
        [
            {
                "next_thought": "All outputs are ready.",
                "next_tool_name": "finish",
                "next_tool_args": {
                    "count": 7,
                    "event": '{"name": "Science Fair", "year": 2026}',
                },
            }
        ]
    )
    with dspy.context(lm=lm):
        outputs = await react.acall(question="Describe the event")

    assert outputs.count == 7
    assert outputs.event == Event(name="Science Fair", year=2026)
    assert len(lm.history) == 1
    assert outputs.trajectory["observation_0"] == "Completed."


@pytest.mark.asyncio
async def test_async_finish_with_explicit_none_for_optional_output_skips_extract():
    class OptionalSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: Optional[str] = dspy.OutputField()  # noqa: UP045 - the typing.Optional spelling is what this test covers

    react = dspy.ReAct(OptionalSignature, tools=[])
    lm = DummyLM(
        [
            {"next_thought": "There is no answer.", "next_tool_name": "finish", "next_tool_args": {"answer": None}},
            {"reasoning": "Extracted.", "answer": "EXTRACT_FALLBACK"},
        ]
    )
    with dspy.context(lm=lm):
        outputs = await react.acall(question="q")

    assert outputs.answer is None
    assert len(lm.history) == 1


@pytest.mark.asyncio
async def test_async_finish_with_explicit_none_for_union_none_output_skips_extract():
    class NullableSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str | None = dspy.OutputField()

    react = dspy.ReAct(NullableSignature, tools=[])
    lm = DummyLM(
        [
            {"next_thought": "There is no answer.", "next_tool_name": "finish", "next_tool_args": {"answer": None}},
            {"reasoning": "Extracted.", "answer": "EXTRACT_FALLBACK"},
        ]
    )
    with dspy.context(lm=lm):
        outputs = await react.acall(question="q")

    assert outputs.answer is None
    assert len(lm.history) == 1


@pytest.mark.asyncio
async def test_async_finish_with_empty_args_falls_back_to_extract():
    react = dspy.ReAct("question -> answer", tools=[])
    lm = DummyLM(
        [
            {"next_thought": "I know the answer.", "next_tool_name": "finish", "next_tool_args": {}},
            {"reasoning": "Extracted reasoning.", "answer": "extracted answer"},
        ]
    )
    with dspy.context(lm=lm):
        outputs = await react.acall(question="q")

    assert outputs.answer == "extracted answer"
    assert outputs.reasoning == "Extracted reasoning."
    assert len(lm.history) == 2
    assert outputs.trajectory["observation_0"] == "Completed."


@pytest.mark.asyncio
async def test_async_error_retry():
    # A tiny tool that always fails
    async def foo(a, b):
        raise Exception("tool error")

    react = dspy.ReAct("a, b -> c:int", tools=[foo])
    lm = DummyLM(
        [
            {
                "next_thought": "I need to add two numbers.",
                "next_tool_name": "foo",
                "next_tool_args": {"a": 1, "b": 2},
            },
            {
                "next_thought": "I need to add two numbers.",
                "next_tool_name": "foo",
                "next_tool_args": {"a": 1, "b": 2},
            },
            # (The model *would* succeed on the 3rd turn, but max_iters=2 stops earlier.)
            {"reasoning": "I added the numbers successfully", "c": 3},
        ]
    )
    with dspy.context(lm=lm):
        outputs = await react.acall(a=1, b=2, max_iters=2)
    traj = outputs.trajectory

    # Exact-match checks (thoughts + tool calls)
    control_expected = {
        "thought_0": "I need to add two numbers.",
        "tool_name_0": "foo",
        "tool_args_0": {"a": 1, "b": 2},
        "thought_1": "I need to add two numbers.",
        "tool_name_1": "foo",
        "tool_args_1": {"a": 1, "b": 2},
    }
    for k, v in control_expected.items():
        assert traj[k] == v, f"{k} mismatch"

    # Flexible checks for observations
    # We only care that each observation mentions our error string; we ignore
    # any extra traceback detail or differing prefixes.
    for i in range(2):
        obs = traj[f"observation_{i}"]
        assert re.search(r"\btool error\b", obs), f"unexpected observation_{i!r}: {obs}"
