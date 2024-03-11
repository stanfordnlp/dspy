import dspy
from dspy import Predict, Signature
from dspy.backends.json import JSONBackend
from dspy.utils.dummies import DummyLanguageModel
from dspy.backends import TemplateBackend


def test_initialization_with_string_signature():
    signature_string = "input1, input2 -> output"
    predict = Predict(signature_string)
    expected_instruction = (
        "Given the fields `input1`, `input2`, produce the fields `output`."
    )
    assert predict.signature.instructions == expected_instruction
    assert predict.signature.instructions == Signature(signature_string).instructions


def test_reset_method():
    predict_instance = Predict("input -> output")
    predict_instance.backend = "modified"
    predict_instance.traces = ["trace"]
    predict_instance.train = ["train"]
    predict_instance.demos = ["demo"]
    predict_instance.reset()
    assert predict_instance.backend is None
    assert predict_instance.traces == []
    assert predict_instance.train == []
    assert predict_instance.demos == []


def test_dump_and_load_state():
    predict_instance = Predict("input -> output")
    predict_instance.backend = "backend_state"
    dumped_state = predict_instance.dump_state()
    new_instance = Predict("input -> output")
    new_instance.load_state(dumped_state)
    assert new_instance.backend == "backend_state"


def test_call_method():
    predict_instance = Predict("input -> output")

    lm = DummyLanguageModel(answers=[["test output"]])
    backend = TemplateBackend(lm=lm)
    dspy.settings.configure(backend=backend)

    result = predict_instance(input="test input")
    assert result.output == "test output"


def test_dump_load_state():
    predict_instance = Predict(Signature("input -> output", "original instructions"))
    dumped_state = predict_instance.dump_state()
    new_instance = Predict(Signature("input -> output", "new instructions"))
    new_instance.load_state(dumped_state)
    assert new_instance.signature.instructions == "original instructions"


def test_forward_method():
    lm = DummyLanguageModel(answers=[["No more responses"]])
    backend = TemplateBackend(lm=lm)
    dspy.settings.configure(backend=backend)

    program = Predict("question -> answer")
    result = program(question="What is 1+1?").answer
    assert result == "No more responses"


def test_forward_method2():
    lm = DummyLanguageModel(
        answers=[[" my first answer\n\nAnswer 2: my second answer"]]
    )
    backend = TemplateBackend(lm=lm)
    dspy.settings.configure(backend=backend)

    program = Predict("question -> answer1, answer2")
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

    lm = DummyLanguageModel(answers=[["my first answer", "my second answer"]])
    backend = TemplateBackend(lm=lm)
    dspy.settings.configure(backend=backend)

    results = program(question="What is 1+1?")
    assert results.completions[0].answer == "my first answer"
    assert results.completions[1].answer == "my second answer"


def test_multi_output_json():
    program = Predict("question -> answer", n=2)

    lm = DummyLanguageModel(
        answers=[
            [
                """{"answer": "my first answer"}""",
                """{"answer": "my second answer"}""",
            ]
        ]
    )
    backend = JSONBackend(lm=lm)
    dspy.settings.configure(backend=backend)

    results = program(question="What is 1+1?")
    assert results.completions[1].answer == "my second answer"
