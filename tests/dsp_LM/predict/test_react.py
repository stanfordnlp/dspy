from dataclasses import dataclass

import pytest

import dspy
from dspy.utils.dummies import DummyLM, dummy_rm


def test_example_no_tools():
    # Create a simple dataset which the model will use with the Retrieve tool.
    lm = DummyLM(
        [
            {
                "Thought_1": "Initial thoughts",
                "Action_1": "Finish[blue]",
            },
        ]
    )
    dspy.settings.configure(lm=lm, rm=dummy_rm())

    program = dspy.ReAct("question -> answer")

    # Check default tools
    assert isinstance(program.tools["Finish"], dspy.Example)

    # Call the ReAct module on a particular input
    question = "What is the color of the sky?"
    result = program(question=question)
    assert result.answer == "blue"


@pytest.mark.asyncio
async def test_async_example_no_tools():
    lm = DummyLM(
        [
            {
                "Thought_1": "Initial thoughts",
                "Action_1": "Finish[blue]",
            },
        ]
    )
    dspy.settings.configure(lm=lm, rm=dummy_rm(), async_mode=True)

    program = dspy.ReAct("question -> answer")

    # Check default tools
    assert isinstance(program.tools["Finish"], dspy.Example)

    # Call the ReAct module on a particular input
    question = "What is the color of the sky?"
    result = await program(question=question)
    assert result.answer == "blue"


def test_example_search():
    # Createa a simple dataset which the model will use with the Retrieve tool.
    lm = DummyLM(
        [
            {
                "Thought_1": "Initial thoughts",
                "Action_1": "Search[the color of the sky]",
            },
            {
                "Thought_2": "More thoughts",
                "Action_2": "Finish[blue]",
            },
        ]
    )
    rm = dummy_rm(
        [
            "We all know the color of the sky is blue.",
            "Somethng about the sky colors",
            "This sentence is completely irellevant to answer the question.",
            "Let's add some more sentences to act as summy passages.",
            "Let's add some more sentences to act as summy passages.",
            "Let's add some more sentences to act as summy passages.",
        ]
    )
    dspy.settings.configure(lm=lm, rm=rm)

    program = dspy.ReAct("question -> answer")

    # Check default tools
    assert len(program.tools) == 2
    assert isinstance(program.tools["Search"], dspy.Retrieve)
    assert isinstance(program.tools["Finish"], dspy.Example)

    # Call the ReAct module on a particular input
    question = "What is the color of the sky?"
    result = program(question=question)
    assert result.answer == "blue"


@pytest.mark.asyncio
async def test_async_example_search():
    lm = DummyLM(
        [
            {
                "Thought_1": "Initial thoughts",
                "Action_1": "Search[the color of the sky]",
            },
            {
                "Thought_2": "More thoughts",
                "Action_2": "Finish[blue]",
            },
        ]
    )
    rm = dummy_rm(
        [
            "We all know the color of the sky is blue.",
            "Somethng about the sky colors",
            "This sentence is completely irellevant to answer the question.",
            "Let's add some more sentences to act as summy passages.",
            "Let's add some more sentences to act as summy passages.",
            "Let's add some more sentences to act as summy passages.",
        ]
    )
    dspy.settings.configure(lm=lm, rm=rm, async_mode=True)

    program = dspy.ReAct("question -> answer")

    # Check default tools
    assert len(program.tools) == 2
    assert isinstance(program.tools["Search"], dspy.Retrieve)
    assert isinstance(program.tools["Finish"], dspy.Example)

    # Call the ReAct module on a particular input
    question = "What is the color of the sky?"
    result = await program(question=question)
    assert result.answer == "blue"


class DummyTool1:
    name = "Tool1"
    input_variable = "query"
    desc = ""
    num_calls = 0

    def __call__(self, *args, **kwargs):
        # test case with no passages attribute
        assert args[0] == "foo"
        self.num_calls += 1
        return "tool 1 output"


@dataclass
class DummyOutput:
    passages: str


class DummyTool2:
    name = "Tool2"
    input_variable = "query"
    desc = ""
    num_calls = 0

    def __call__(self, *args, **kwargs):
        # test case with passages attribute
        assert args[0] == "bar"
        self.num_calls += 1
        return DummyOutput(passages="tool 2 output")


def test_custom_tools():
    lm = DummyLM(
        [
            {
                "Thought_1": "Initial thoughts",
                "Action_1": "Tool1[foo]",
            },
            {
                "Thought_2": "More thoughts",
                "Action_2": "Tool2[bar]",
            },
            {
                "Thought_3": "Even more thoughts",
                "Action_3": "Finish[baz]",
            },
        ]
    )
    dspy.settings.configure(lm=lm)

    tool1 = DummyTool1()
    tool2 = DummyTool2()
    program = dspy.ReAct("question -> answer", tools=[tool1, tool2])

    question = "What is the color of the sky?"
    result = program(question=question)
    assert result.answer == "baz"

    # each tool should be called only once
    assert tool1.num_calls == 1
    assert tool2.num_calls == 1


@pytest.mark.asyncio
async def test_async_custom_tools():
    lm = DummyLM(
        [
            {
                "Thought_1": "Initial thoughts",
                "Action_1": "Tool1[foo]",
            },
            {
                "Thought_2": "More thoughts",
                "Action_2": "Tool2[bar]",
            },
            {
                "Thought_3": "Even more thoughts",
                "Action_3": "Finish[baz]",
            },
        ]
    )
    dspy.settings.configure(lm=lm, async_mode=True)

    tool1 = DummyTool1()
    tool2 = DummyTool2()
    program = dspy.ReAct("question -> answer", tools=[tool1, tool2])

    question = "What is the color of the sky?"
    result = await program(question=question)
    assert result.answer == "baz"

    # each tool should be called only once
    assert tool1.num_calls == 1
    assert tool2.num_calls == 1
