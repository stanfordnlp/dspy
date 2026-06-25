import pytest

import dspy
from dspy import Example


def test_example_initialization():
    example = Example(a=1, b=2)
    assert example.a == 1
    assert example.b == 2


def test_example_initialization_from_base():
    base = Example(a=1, b=2)
    example = Example(base=base, c=3)
    assert example.a == 1
    assert example.b == 2
    assert example.c == 3


def test_example_initialization_from_dict():
    base_dict = {"a": 1, "b": 2}
    example = Example(base=base_dict, c=3)
    assert example.a == 1
    assert example.b == 2
    assert example.c == 3


def test_example_set_get_item():
    example = Example()
    example["a"] = 1
    assert example["a"] == 1


def test_example_attribute_access():
    example = Example(a=1)
    assert example.a == 1
    example.a = 2
    assert example.a == 2


def test_example_deletion():
    example = Example(a=1, b=2)
    del example["a"]
    with pytest.raises(AttributeError):
        _ = example.a


def test_example_len():
    example = Example(a=1, b=2, dspy_hidden=3)
    assert len(example) == 2


def test_example_repr_str_img():
    example = Example(
        img=dspy.Image(url="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")
    )
    assert (
        repr(example)
        == "Example({'img': Image(url=data:image/gif;base64,<IMAGE_BASE_64_ENCODED(56)>)}) (input_keys=None)"
    )
    assert (
        str(example)
        == "Example({'img': Image(url=data:image/gif;base64,<IMAGE_BASE_64_ENCODED(56)>)}) (input_keys=None)"
    )


def test_example_repr_str():
    example = Example(a=1)
    assert repr(example) == "Example({'a': 1}) (input_keys=None)"
    assert str(example) == "Example({'a': 1}) (input_keys=None)"


def test_example_eq():
    example1 = Example(a=1, b=2)
    example2 = Example(a=1, b=2)
    assert example1 == example2
    assert example1 != ""


def test_example_hash():
    example1 = Example(a=1, b=2)
    example2 = Example(a=1, b=2)
    assert hash(example1) == hash(example2)


def test_example_keys_values_items():
    example = Example(a=1, b=2, dspy_hidden=3)
    assert set(example.keys()) == {"a", "b"}
    assert 1 in example.values()
    assert ("b", 2) in example.items()


def test_example_get():
    example = Example(a=1, b=2)
    assert example.get("a") == 1
    assert example.get("c", "default") == "default"


def test_example_with_inputs():
    example = Example(a=1, b=2).with_inputs("a")
    assert example._input_keys == {"a"}


def test_example_inputs_labels():
    example = Example(a=1, b=2).with_inputs("a")
    inputs = example.inputs()
    assert inputs.toDict() == {"a": 1}
    labels = example.labels()
    assert labels.toDict() == {"b": 2}


def test_example_copy_without():
    example = Example(a=1, b=2)
    copied = example.copy(c=3)
    assert copied.a == 1
    assert copied.c == 3
    without_a = copied.without("a")
    with pytest.raises(AttributeError):
        _ = without_a.a


def test_example_copy_preserves_input_keys():
    """copy()/without() must preserve the input/label split.

    Regression: the input keys were reset to None on copy, so .inputs()/.labels()
    raised on any copied Example (and Example(base=other) lost the split too).
    """
    example = Example(question="q", answer="a").with_inputs("question")

    copied = example.copy(answer="b")
    assert copied._input_keys == {"question"}
    assert copied.inputs().toDict() == {"question": "q"}
    assert copied.labels().toDict() == {"answer": "b"}

    # without() routes through copy(); the split must survive for remaining fields.
    no_extra = example.copy(source="web").without("source")
    assert no_extra._input_keys == {"question"}
    assert no_extra.inputs().toDict() == {"question": "q"}

    # Constructing directly from an Example base also preserves the split.
    assert Example(base=example)._input_keys == {"question"}


def test_prediction_copy_does_not_require_input_keys():
    # Example subclasses (e.g. Prediction) don't keep _input_keys; copy() must not crash.
    import dspy

    copied = dspy.Prediction(answer="a").copy(answer="b")
    assert copied.answer == "b"


def test_example_to_dict():
    example = Example(a=1, b=2)
    assert example.toDict() == {"a": 1, "b": 2}


def test_example_to_dict_with_history():
    """Test that Example.toDict() properly serializes dspy.History objects."""
    history = dspy.History(
        messages=[
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is the capital of Germany?", "answer": "Berlin"},
        ]
    )
    example = Example(question="Test question", history=history, answer="Test answer")

    result = example.toDict()

    # Verify the result is a dictionary
    assert isinstance(result, dict)
    assert "history" in result

    # Verify history is serialized to a dict (not a History object)
    assert isinstance(result["history"], dict)
    assert "messages" in result["history"]
    assert result["history"]["messages"] == [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "What is the capital of Germany?", "answer": "Berlin"},
    ]

    # Verify JSON serialization works
    import json
    json_str = json.dumps(result)
    restored = json.loads(json_str)
    assert restored["history"]["messages"] == result["history"]["messages"]
