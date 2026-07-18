import re

import pytest
from pydantic import BaseModel

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


def test_react_recovers_from_adapter_parse_error():
    # Regression test for https://github.com/stanfordnlp/dspy/issues/8377:
    # the LM aperiodically emits a wrong section header (e.g. `tool_args`
    # instead of `next_tool_args`); ReAct should feed the parse failure back
    # into the trajectory and let the model self-correct, mirroring how tool
    # execution errors are handled, instead of crashing the whole call.
    def add(a: int, b: int) -> int:
        return a + b

    react = dspy.ReAct("a, b -> c:int", tools=[add])
    lm = DummyLM(
        [
            # Wrong field name, exactly as reported in #8377 -> AdapterParseError.
            {
                "next_thought": "I need to add two numbers.",
                "next_tool_name": "add",
                "tool_args": {"a": 1, "b": 2},
            },
            # The model self-corrects on the next iteration.
            {
                "next_thought": "Retrying with the correct fields.",
                "next_tool_name": "add",
                "next_tool_args": {"a": 1, "b": 2},
            },
            {
                "next_thought": "I have the sum, finishing.",
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            {"reasoning": "Added the numbers.", "c": "3"},
        ]
    )
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False))

    outputs = react(a=1, b=2)
    traj = outputs.trajectory

    assert outputs.c == 3
    # The parse failure was recorded as a full 4-key step (preserving the
    # trajectory shape truncate_trajectory relies on), reusing the fields that
    # did parse and pointing out what was missing.
    assert "could not be parsed" in traj["observation_0"]
    assert "next_tool_args" in traj["observation_0"]
    assert traj["thought_0"] == "I need to add two numbers."
    assert traj["tool_name_0"] == "add"
    assert traj["tool_args_0"] == {}
    # ...and the loop continued: the next iteration executed the tool normally.
    assert traj["tool_name_1"] == "add"
    assert traj["observation_1"] == 3
    assert traj["tool_name_2"] == "finish"


def test_react_parse_error_every_iteration_falls_back_to_extract():
    def add(a: int, b: int) -> int:
        return a + b

    max_iters = 3
    react = dspy.ReAct("a, b -> c:int", tools=[add])
    lm = DummyLM(
        [
            # Every iteration produces an unparseable step.
            {
                "next_thought": "I need to add two numbers.",
                "next_tool_name": "add",
                "tool_args": {"a": 1, "b": 2},
            },
        ]
        * max_iters
        + [{"reasoning": "Could not run any tool.", "c": "3"}]
    )
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False))

    outputs = react(a=1, b=2, max_iters=max_iters)
    traj = outputs.trajectory

    # No crash: the parse failure of every iteration is in the trajectory and
    # the extract fallback still produced the outputs.
    assert outputs.c == 3
    for idx in range(max_iters):
        assert "could not be parsed" in traj[f"observation_{idx}"]


@pytest.mark.asyncio
async def test_async_react_recovers_from_adapter_parse_error():
    async def add(a: int, b: int) -> int:
        return a + b

    react = dspy.ReAct("a, b -> c:int", tools=[add])
    lm = DummyLM(
        [
            {
                "next_thought": "I need to add two numbers.",
                "next_tool_name": "add",
                "tool_args": {"a": 1, "b": 2},
            },
            {
                "next_thought": "Retrying with the correct fields.",
                "next_tool_name": "add",
                "next_tool_args": {"a": 1, "b": 2},
            },
            {
                "next_thought": "I have the sum, finishing.",
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            {"reasoning": "Added the numbers.", "c": "3"},
        ]
    )
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False))

    outputs = await react.acall(a=1, b=2)
    traj = outputs.trajectory

    assert outputs.c == 3
    assert "could not be parsed" in traj["observation_0"]
    assert traj["tool_name_1"] == "add"
    assert traj["observation_1"] == 3


def test_react_parse_failure_preserves_trajectory_truncation_alignment():
    # A parse-failure step must contribute the same four keys as a normal step,
    # so truncate_trajectory (which pops keys in groups of four) stays aligned.
    def add(a: int, b: int) -> int:
        return a + b

    react = dspy.ReAct("a, b -> c:int", tools=[add])
    lm = DummyLM(
        [
            {
                "next_thought": "I need to add two numbers.",
                "next_tool_name": "add",
                "tool_args": {"a": 1, "b": 2},
            },
            {
                "next_thought": "Retrying with the correct fields.",
                "next_tool_name": "add",
                "next_tool_args": {"a": 1, "b": 2},
            },
            {
                "next_thought": "I have the sum, finishing.",
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            {"reasoning": "Added the numbers.", "c": "3"},
        ]
    )
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False))

    traj = react(a=1, b=2).trajectory
    assert len(traj) % 4 == 0

    # Truncating drops the oldest step (the parse failure) as one whole unit,
    # leaving the first real tool call intact and aligned.
    truncated = react.truncate_trajectory(dict(traj))
    assert truncated["thought_1"] == "Retrying with the correct fields."
    assert truncated["tool_name_1"] == "add"
    assert truncated["observation_1"] == 3


def test_react_extract_parse_error_is_retried_with_feedback():
    def add(a: int, b: int) -> int:
        return a + b

    react = dspy.ReAct("a, b -> c:int", tools=[add])
    lm = DummyLM(
        [
            {
                "next_thought": "I have the answer, finishing.",
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            # The extraction step emits a wrong field name -> AdapterParseError.
            {"reasoning": "Adding the numbers.", "d": "3"},
            # The retry (with parse feedback in the prompt) self-corrects.
            {"reasoning": "Adding the numbers.", "c": "3"},
        ]
    )
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False))

    outputs = react(a=1, b=2)
    assert outputs.c == 3
    # The prompt-only feedback is not stored in the returned trajectory.
    assert "parse_feedback" not in outputs.trajectory


def test_react_extract_parse_error_propagates_after_retry():
    from dspy.utils.exceptions import AdapterParseError

    def add(a: int, b: int) -> int:
        return a + b

    react = dspy.ReAct("a, b -> c:int", tools=[add])
    lm = DummyLM(
        [
            {
                "next_thought": "I have the answer, finishing.",
                "next_tool_name": "finish",
                "next_tool_args": {},
            },
            # The extraction step fails to parse twice: the error is real, surface it.
            {"reasoning": "Adding the numbers.", "d": "3"},
            {"reasoning": "Adding the numbers.", "d": "3"},
        ]
    )
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False))

    with pytest.raises(AdapterParseError):
        react(a=1, b=2)


def test_codeact_recovers_from_adapter_parse_error():
    from dspy.predict import CodeAct

    # No tools: keeps the test free of the Deno interpreter (tool registration
    # is the only pre-loop interpreter use), while still exercising the loop.
    program = CodeAct("question -> answer", tools=[], max_iters=2)
    lm = DummyLM(
        [
            # Both iterations emit an unparseable step (wrong field name).
            {"reasoning": "Thinking.", "code": "print(42)"},
            {"reasoning": "Thinking.", "code": "print(42)"},
            {"reasoning": "Done.", "answer": "42"},
        ]
    )
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter(use_json_adapter_fallback=False))

    outputs = program(question="What is 6*7?")
    assert outputs.answer == "42"
    assert "could not be parsed" in outputs.trajectory["observation_0"]
    assert "could not be parsed" in outputs.trajectory["observation_1"]
