import pytest

import dspy


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("location", ["lm", "predictor", "call"])
@pytest.mark.parametrize("reasoning,effort", [
    ({"enabled": False}, "off"), ({"effort": "none"}, "off"),
    ({"effort": "off"}, "off"),
    ({"effort": "high", "summary": "detailed"}, "high"), ({}, "medium"),
])
def test_cot_reasoning_dictionary(native, location, reasoning, effort):
    from tests.test_utils.engines import make_response

    requests = []

    class Engine:
        supports_reasoning = native

        def complete(self, request):
            requests.append(request)
            plan = "<plan>explicit plan</plan>" if "<plan>" in request.system else ""
            return make_response(plan + "<answer>42</answer>", thinking="native summary")

    options = {"reasoning": reasoning}
    lm = dspy.LM("custom/reasoning", engine=Engine(), cache=False, **(options if location == "lm" else {}))
    cot = dspy.ChainOfThought("question -> answer", **(options if location == "predictor" else {}))
    with dspy.context(lm=lm, adapter=dspy.XMLAdapter(use_json_adapter_fallback=False)):
        result = cot(question="19 + 23", config=options if location == "call" else {})
    assert result.answer == "42"
    assert result.reasoning == ("native summary" if native and effort != "off" else "explicit plan")
    assert len(requests) == 1
    assert requests[0].config.reasoning.effort == effort
    if "summary" in reasoning:
        assert requests[0].config.reasoning.summary == "detailed"


@pytest.mark.parametrize("defaults,override,effort", [
    ({"reasoning": {"effort": "high"}}, {"reasoning_effort": "none"}, "off"),
    ({"reasoning_effort": "high"}, {"reasoning": {"enabled": False}}, "off"),
    ({"reasoning": {"enabled": False}}, {"reasoning_effort": "high"}, "high"),
    ({"reasoning_effort": "none"}, {"reasoning": {"effort": "high"}}, "high"),
])
def test_cot_reasoning_alias_override(defaults, override, effort):
    from tests.test_utils.engines import make_response, recording_lm

    text = "<plan>explicit plan</plan><answer>42</answer>" if effort == "off" else "<answer>42</answer>"
    lm = recording_lm([make_response(text, thinking="native summary")], **defaults)
    with dspy.context(lm=lm, adapter=dspy.XMLAdapter(use_json_adapter_fallback=False)):
        result = dspy.ChainOfThought("question -> answer")(question="19 + 23", config=override)
    assert result.reasoning == ("explicit plan" if effort == "off" else "native summary")
    assert lm.engine.requests[0].config.reasoning.effort == effort


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

    cot = dspy.ChainOfThought("question -> answer", rationale_field_type=str)
    result = cot(question="What is the answer?")

    # Test that we can use string methods on result.reasoning
    assert isinstance(result.reasoning, str)
    assert result.reasoning.strip() == "Let me think step by step"
    assert result.reasoning.lower() == "let me think step by step"
    assert "step by step" in result.reasoning
    assert len(result.reasoning) == 25


def test_reasoning_error_message():
    reasoning = dspy.Reasoning(content="Hello")

    with pytest.raises(AttributeError, match="`Reasoning` object has no attribute 'nonexistent_method'"):
        reasoning.nonexistent_method


def test_native_reasoning_default_survives_provider_omission():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
        reasoning: dspy.Reasoning = dspy.OutputField(default_factory=lambda: dspy.Reasoning(content="(omitted)"))

    adapter = dspy.ChatAdapter()
    # Native reasoning strips the field from the signature the LM formats against; the provider
    # then omits native reasoning entirely.
    processed = Sig.delete("reasoning")
    from tests.test_utils.engines import make_response

    responses = [make_response("[[ ## answer ## ]]\n42\n\n[[ ## completed ## ]]")]

    value = adapter._call_postprocess(processed, Sig, responses)[0]

    assert value["answer"] == "42"
    assert value["reasoning"] == dspy.Reasoning(content="(omitted)")
