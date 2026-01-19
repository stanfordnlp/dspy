import pytest

import dspy


def test_reasoning_basic_operations():
    reasoning = dspy.Reasoning(content="Hello World")

    # Test str conversion
    assert str(reasoning) == "Hello World"
    assert repr(reasoning) == "'Hello World'"

    # Test equality
    assert reasoning == "Hello World"
    assert reasoning == dspy.Reasoning(content="Hello World")
    assert reasoning != "hello world"
    assert reasoning != dspy.Reasoning(content="hello world")

    # Test len
    assert len(reasoning) == 11

    # Test indexing
    assert reasoning[0] == "H"
    assert reasoning[-1] == "d"
    assert reasoning[0:5] == "Hello"

    # Test in operator
    assert "World" in reasoning
    assert "xyz" not in reasoning

    # Test iteration
    chars = [c for c in reasoning]
    assert len(chars) == 11
    assert chars[0] == "H"


def test_reasoning_concatenation():
    reasoning = dspy.Reasoning(content="Hello")

    # Test + operator
    result1 = reasoning + " World"
    assert result1 == "Hello World"
    assert isinstance(result1, str)

    # Test reverse + operator
    result2 = "Prefix: " + reasoning
    assert result2 == "Prefix: Hello"
    assert isinstance(result2, str)

    # Test Reasoning + Reasoning
    reasoning2 = dspy.Reasoning(content=" World")
    result3 = reasoning + reasoning2
    assert isinstance(result3, dspy.Reasoning)
    assert result3.content == "Hello World"


def test_reasoning_string_methods():
    reasoning = dspy.Reasoning(content="  Hello World  ")

    # Test strip
    assert reasoning.strip() == "Hello World"

    # Test lower/upper
    assert reasoning.lower() == "  hello world  "
    assert reasoning.upper() == "  HELLO WORLD  "

    # Test split
    assert reasoning.strip().split() == ["Hello", "World"]
    assert reasoning.strip().split(" ") == ["Hello", "World"]

    # Test replace
    assert reasoning.replace("World", "Python") == "  Hello Python  "

    # Test startswith/endswith
    assert reasoning.strip().startswith("Hello")
    assert reasoning.strip().endswith("World")
    assert not reasoning.strip().startswith("World")

    # Test find
    assert reasoning.find("World") == 8
    assert reasoning.find("xyz") == -1

    # Test count
    assert reasoning.count("l") == 3

    # Test join
    assert reasoning.strip().join(["a", "b", "c"]) == "aHello WorldbHello Worldc"


def test_reasoning_with_chain_of_thought():
    from dspy.utils import DummyLM

    lm = DummyLM([{"reasoning": "Let me think step by step", "answer": "42"}])
    dspy.configure(lm=lm)

    cot = dspy.ChainOfThought("question -> answer")
    result = cot(question="What is the answer?")

    # Test that we can use string methods on result.reasoning
    assert isinstance(result.reasoning, dspy.Reasoning)
    assert result.reasoning.strip() == "Let me think step by step"
    assert result.reasoning.lower() == "let me think step by step"
    assert "step by step" in result.reasoning
    assert len(result.reasoning) == 25


def test_reasoning_error_message():
    reasoning = dspy.Reasoning(content="Hello")

    with pytest.raises(AttributeError, match="`Reasoning` object has no attribute 'nonexistent_method'"):
        reasoning.nonexistent_method
