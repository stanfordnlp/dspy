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


def test_example_hash_is_order_insensitive():
    # `__eq__` compares the underlying dict (order-insensitive), so the hash
    # contract requires `__hash__` to be order-insensitive as well.
    example1 = Example(a=1, b=2)
    example2 = Example(b=2, a=1)
    assert example1 == example2
    assert hash(example1) == hash(example2)


def test_example_set_and_dict_lookup_after_reorder():
    # Direct consequence of the hash contract: equal Examples constructed in
    # different field orders must deduplicate in sets and look up in dicts.
    example1 = Example(a=1, b=2)
    example2 = Example(b=2, a=1)
    assert len({example1, example2}) == 1
    assert {example1: "v"}.get(example2) == "v"


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


def test_assigning_to_a_method_named_field_updates_the_store():
    """A field whose name shadows a method must not hold two different values.

    Assignment previously wrote an instance attribute and left `_store` alone, so
    `ex.items` and `ex["items"]` diverged.
    """
    ex = dspy.Example(items=["a"])
    assert ex["items"] == ["a"]

    ex.items = ["b"]

    assert ex["items"] == ["b"]
    assert ex.toDict()["items"] == ["b"]


def test_method_named_field_keeps_the_method_callable():
    """Fields never shadow the mapping API, which other code and users rely on."""
    ex = dspy.Example(items=["a"], keys=["k"], values=["v"], get=1)

    assert callable(ex.items)
    assert sorted(ex.keys()) == ["get", "items", "keys", "values"]
    assert dict(ex.items())["items"] == ["a"]
    assert ex.get("items") == ["a"]


def test_method_named_field_warns_at_construction(caplog):
    from dspy.primitives.example import _WARNED_SHADOWED_FIELDS

    _WARNED_SHADOWED_FIELDS.clear()
    try:
        with caplog.at_level("WARNING", logger="dspy.primitives.example"):
            dspy.Example(items=["a"])

        assert "items" in caplog.text
        assert "subscript" in caplog.text
    finally:
        _WARNED_SHADOWED_FIELDS.clear()


def test_ordinary_field_assignment_is_unchanged():
    ex = dspy.Example(question="q", answer="a")
    ex.answer = "b"

    assert ex.answer == "b"
    assert ex["answer"] == "b"
    assert "answer" not in ex.__dict__


def test_private_attributes_still_bypass_the_store():
    ex = dspy.Example(question="q")
    ex._input_keys = {"question"}

    assert ex._input_keys == {"question"}
    assert "_input_keys" not in ex.keys(include_dspy=True)


def test_non_string_keys_are_accepted():
    """A dict passed as `base` may hold keys of any hashable type."""
    ex = dspy.Example(base={1: "value", "question": "q"})

    assert ex[1] == "value"
    assert ex.question == "q"


def test_shadowing_warning_is_emitted_once_per_field(caplog):
    """copy()/without()/with_inputs() each build a new instance; one warning is enough."""
    from dspy.primitives.example import _WARNED_SHADOWED_FIELDS

    _WARNED_SHADOWED_FIELDS.clear()
    try:
        with caplog.at_level("WARNING", logger="dspy.primitives.example"):
            ex = dspy.Example(items=["a"])
            for _ in range(5):
                ex.copy()

        assert caplog.text.count("share a name with a method") == 1
    finally:
        _WARNED_SHADOWED_FIELDS.clear()
