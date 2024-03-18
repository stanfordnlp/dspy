import dsp
import dspy
from dspy import Predict, Signature
from dspy.backends.json import JSONBackend
from dspy.backends import TemplateBackend
from dspy.utils.dummies import DummyLM, DummyLanguageModel
import copy
import textwrap


def test_initialization_with_string_signature():
    signature_string = "input1, input2 -> output"
    predict = Predict(signature_string)
    expected_instruction = (
        "Given the fields `input1`, `input2`, produce the fields `output`."
    )
    assert predict.signature.instructions == expected_instruction
    assert predict.signature.instructions == Signature(signature_string).instructions


def test_reset_method():
    dsp.settings.get("experimental", False)

    predict_instance = Predict("input -> output")
    predict_instance.lm = "modified"

    predict_instance.traces = ["trace"]
    predict_instance.train = ["train"]
    predict_instance.demos = ["demo"]
    predict_instance.reset()
    assert predict_instance.lm is None
    assert predict_instance.traces == []
    assert predict_instance.train == []
    assert predict_instance.demos == []


def test_dump_and_load_state():
    predict_instance = Predict("input -> output")
    predict_instance.lm = "lm_state"
    dumped_state = predict_instance.dump_state()
    new_instance = Predict("input -> output")
    new_instance.load_state(dumped_state)
    assert new_instance.lm == "lm_state"


def test_call_method():
    predict_instance = Predict("input -> output")
    lm = DummyLM(["test output"])
    dspy.settings.configure(lm=lm)
    result = predict_instance(input="test input")
    assert result.output == "test output"
    assert lm.get_convo(-1) == (
        "Given the fields `input`, produce the fields `output`.\n"
        "\n---\n\n"
        "Follow the following format.\n\n"
        "Input: ${input}\n"
        "Output: ${output}\n"
        "\n---\n\n"
        "Input: test input\n"
        "Output: test output"
    )


def test_call_method_experimental():
    dspy.settings.configure(experimental=True)
    predict_instance = Predict("input -> output")

    lm = DummyLanguageModel(answers=[["test output"]])
    backend = TemplateBackend(lm=lm)
    dspy.settings.configure(backend=backend, lm=None)

    result = predict_instance(input="test input")
    assert result.output == "test output"


def test_dump_load_state():
    dspy.settings.configure(experimental=False)

    predict_instance = Predict(Signature("input -> output", "original instructions"))
    dumped_state = predict_instance.dump_state()
    new_instance = Predict(Signature("input -> output", "new instructions"))
    new_instance.load_state(dumped_state)
    assert new_instance.signature.instructions == "original instructions"


def test_forward_method():
    dspy.settings.configure(experimental=False)

    program = Predict("question -> answer")
    dspy.settings.configure(lm=DummyLM([]), backend=None)
    result = program(question="What is 1+1?").answer
    assert result == "No more responses"


def test_forward_method_experimental():
    dspy.settings.configure(experimental=True)

    lm = DummyLanguageModel(answers=[["No more responses"]])
    backend = TemplateBackend(lm=lm)
    dspy.settings.configure(backend=backend, lm=None)

    program = Predict("question -> answer")
    result = program(question="What is 1+1?").answer
    assert result == "No more responses"


def test_forward_method2():
    dspy.settings.configure(experimental=False)

    program = Predict("question -> answer1, answer2")
    dspy.settings.configure(lm=DummyLM(["my first answer", "my second answer"]))
    result = program(question="What is 1+1?")
    assert result.answer1 == "my first answer"
    assert result.answer2 == "my second answer"


def test_forward_method2_experimental():
    dspy.settings.configure(experimental=True)

    lm = DummyLanguageModel(
        answers=[[" my first answer\n\nAnswer 2: my second answer"]]
    )
    backend = TemplateBackend(lm=lm)
    dspy.settings.configure(backend=backend, lm=None)

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
    dspy.settings.configure(experimental=False)

    program = Predict("question -> answer", n=2)
    dspy.settings.configure(
        lm=DummyLM(["my first answer", "my second answer"]), backend=False
    )
    results = program(question="What is 1+1?")
    assert results.completions[0].answer == "my first answer"
    assert results.completions[1].answer == "my second answer"


def test_multi_output_experimental():
    dspy.settings.configure(experimental=True)

    program = Predict("question -> answer", n=2)

    lm = DummyLanguageModel(answers=[["my first answer", "my second answer"]])
    backend = TemplateBackend(lm=lm)
    dspy.settings.configure(backend=backend, lm=None)

    results = program(question="What is 1+1?")
    assert results.completions[0].answer == "my first answer"
    assert results.completions[1].answer == "my second answer"


def test_multi_output_json_experimental():
    dspy.settings.configure(experimental=True)

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
    dspy.settings.configure(backend=backend, lm=None)

    results = program(question="What is 1+1?")
    assert results.completions[1].answer == "my second answer"


def test_multi_output2():
    dspy.settings.configure(experimental=False)

    program = Predict("question -> answer1, answer2", n=2)
    dspy.settings.configure(
        lm=DummyLM(
            [
                "my 0 answer\nAnswer 2: my 2 answer",
                "my 1 answer\nAnswer 2: my 3 answer",
            ],
        )
    )
    results = program(question="What is 1+1?")
    assert results.completions[0].answer1 == "my 0 answer"
    assert results.completions[1].answer1 == "my 1 answer"
    assert results.completions[0].answer2 == "my 2 answer"
    assert results.completions[1].answer2 == "my 3 answer"


def test_named_predictors():
    class MyModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.inner = Predict("question -> answer")

    program = MyModule()
    assert program.named_predictors() == [("inner", program.inner)]

    # Check that it also works the second time.
    program2 = copy.deepcopy(program)
    assert program2.named_predictors() == [("inner", program2.inner)]


def test_output_only():
    class OutputOnlySignature(dspy.Signature):
        output = dspy.OutputField()

    predictor = Predict(OutputOnlySignature)

    lm = DummyLM(["short answer"])
    dspy.settings.configure(lm=lm)
    assert predictor().output == "short answer"

    assert lm.get_convo(-1) == textwrap.dedent(
        """\
        Given the fields , produce the fields `output`.
        
        ---
        
        Follow the following format.
        
        Output: ${output}
        
        ---
        
        Output: short answer"""
    )
