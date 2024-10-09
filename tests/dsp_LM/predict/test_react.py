from dataclasses import dataclass

import dspy
from dspy.utils.dummies import DSPDummyLM, dummy_rm


def test_example_no_tools():
    # Create a simple dataset which the model will use with the Retrieve tool.
    lm = DSPDummyLM(
        [
            "Initial thoughts",  # Thought_1
            "Finish[blue]",  # Action_1
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

    # For debugging
    print("---")
    for row in lm.history:
        print(row["prompt"])
        print("Response:", row["response"]["choices"][0]["text"])
        print("---")

    assert lm.get_convo(-1).endswith(
        "Question: What is the color of the sky?\n" "Thought 1: Initial thoughts\n" "Action 1: Finish[blue]"
    )


def test_example_search():
    # Createa a simple dataset which the model will use with the Retrieve tool.
    lm = DSPDummyLM(
        [
            "Initial thoughts",  # Thought_1
            "Search[the color of the sky]",  # Thought_1
            "More thoughts",  # Thought_2
            "Finish[blue]",  # Action_2
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

    # For debugging
    print(lm.get_convo(-1))

    assert lm.get_convo(-1).endswith(
        "Question: What is the color of the sky?\n\n"
        "Thought 1: Initial thoughts\n\n"
        "Action 1: Search[the color of the sky]\n\n"
        "Observation 1:\n"
        "[1] «We all know the color of the sky is blue.»\n"
        "[2] «Somethng about the sky colors»\n"
        "[3] «This sentence is completely irellevant to answer the question.»\n\n"
        "Thought 2: More thoughts\n\n"
        "Action 2: Finish[blue]"
    )


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
    lm = DSPDummyLM(
        [
            "Initial thoughts",
            "Tool1[foo]",
            "More thoughts",
            "Tool2[bar]",
            "Even more thoughts",
            "Finish[baz]",
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
    assert lm.get_convo(-1).endswith(
        "Question: What is the color of the sky?\n\n"
        "Thought 1: Initial thoughts\n\n"
        "Action 1: Tool1[foo]\n\n"
        "Observation 1: tool 1 output\n\n"
        "Thought 2: More thoughts\n\n"
        "Action 2: Tool2[bar]\n\n"
        "Observation 2: tool 2 output\n\n"
        "Thought 3: Even more thoughts\n\n"
        "Action 3: Finish[baz]"
    )
