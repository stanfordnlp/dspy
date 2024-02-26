import dspy
from dspy.utils.dummies import dummy_rm, DummyLM


def test_example_no_tools():
    # Createa a simple dataset which the model will use with the Retrieve tool.
    lm = DummyLM(
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
        "Question: What is the color of the sky?\n"
        "Thought 1: Initial thoughts\n"
        "Action 1: Finish[blue]"
    )


def test_example_search():
    # Createa a simple dataset which the model will use with the Retrieve tool.
    lm = DummyLM(
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
