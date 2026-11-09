from unittest.mock import patch

import pytest

import dspy
from dspy import Example
from dspy.clients import Provider
from dspy.predict import Predict
from dspy.teleprompt import BootstrapFinetune
from dspy.utils.dummies import DummyLM


class RecordingTrainingProvider(Provider):
    """Capture training data locally without launching a remote job."""

    def __init__(self):
        super().__init__()
        self.finetunable = True
        self.data = []

    def finetune(self, job, model, train_data, train_data_format, train_kwargs=None):
        self.data = train_data
        return "custom/fitted"


class TrainingReplayEngine:
    def __init__(self, provider, native=False):
        self.provider = provider
        self.supports_reasoning = native

    def complete(self, request):
        from tests.test_utils.engines import make_response

        if request.model == "custom/fitted":
            text = self.provider.data[0]["messages"][-1]["content"]
        else:
            xml = "<answer>" in request.system
            fields = {"plan_2": "training-plan-73", "plan": "training-plan-73",
                      "reasoning": "training-plan-73", "answer": "learned-91"}
            if "<plan_2>" in request.system or "[[ ## plan_2 ## ]]" in request.system:
                fields["plan"] = "task-plan-19"
            text = "\n".join(
                f"<{name}>{value}</{name}>" if xml else f"[[ ## {name} ## ]]\n{value}"
                for name, value in fields.items()
                if (f"<{name}>" if xml else f"[[ ## {name} ## ]]") in request.system
            )
        return make_response(text, thinking="training-plan-73" if self.supports_reasoning else None)


@pytest.mark.parametrize("adapter_type", [dspy.ChatAdapter, dspy.XMLAdapter])
@pytest.mark.parametrize("mode", [
    "default", "string", "collision", "native-teacher", "native-student", "disabled-student", "disabled-teacher",
])
def test_bootstrap_finetune_target_replays_at_inference(adapter_type, mode):
    provider = RecordingTrainingProvider()
    engine = TrainingReplayEngine(provider, native=mode in ("native-student", "disabled-student", "disabled-teacher"))
    lm = dspy.LM("custom/student", engine=engine, provider=provider, cache=False)
    signature = "question -> plan, answer" if mode == "collision" else "question -> answer"
    student = dspy.ChainOfThought(signature, **({"rationale_field_type": str} if mode == "string" else {}))
    student.set_lm(lm)
    if mode == "disabled-student":
        student.predict.update_config(reasoning_effort="none")
    if mode == "collision":
        student.predict.demos = [dspy.Example(question="demo", reasoning="demo-reasoning-23",
                                            plan="demo-plan-47", answer="demo-answer-59")]
    teacher = None
    if mode in ("native-teacher", "disabled-student", "disabled-teacher"):
        teacher = dspy.ChainOfThought(signature, reasoning_effort="none" if mode == "disabled-teacher" else "high")
        teacher.set_lm(dspy.LM("custom/teacher", engine=TrainingReplayEngine(provider, native=True), cache=False))
    with dspy.context(adapter=adapter_type(use_json_adapter_fallback=False)):
        compiled = dspy.BootstrapFinetune(num_threads=1).compile(
            student, teacher=teacher,
            trainset=[dspy.Example(question="learn this", answer="learned-91").with_inputs("question")],
        )
        assert len(provider.data) == 1
        result = compiled(question="learn this")
    target = provider.data[0]["messages"][-1]["content"]
    expected_fields = {"string": {"reasoning"}, "collision": {"plan", "plan_2"},
                       "native-student": set(), "disabled-teacher": set()}.get(mode, {"plan"})
    for field in ("reasoning", "plan", "plan_2"):
        marker = f"<{field}>" if adapter_type is dspy.XMLAdapter else f"[[ ## {field} ## ]]"
        assert (marker in target) == (field in expected_fields)
    assert result.answer == "learned-91"
    assert str(result.reasoning) == "training-plan-73"
    if mode == "collision":
        assert result.plan == "task-plan-19"
        demo = provider.data[0]["messages"][2]["content"]
        marker = "<plan_2>" if adapter_type is dspy.XMLAdapter else "[[ ## plan_2 ## ]]"
        assert marker in demo and "demo-reasoning-23" in demo and "demo-plan-47" in demo
        assert student.predict.demos[0].reasoning == "demo-reasoning-23"
    assert list(student.predict.signature.output_fields) == ["reasoning", *signature.split(" -> ")[1].split(", ")]


@pytest.mark.parametrize("adapter_type", [dspy.ChatAdapter, dspy.XMLAdapter])
def test_bootstrap_finetune_preserves_tool_supervision(adapter_type):
    from tests.test_utils.engines import RecordingEngine, make_response

    provider = RecordingTrainingProvider()
    engine = RecordingEngine([make_response(tool_calls=[("call-73", "lookup", {"query": "tool-query-91"})])])
    lm = dspy.LM("custom/student", engine=engine, provider=provider, cache=False)
    student = dspy.Predict("question, tools: list[dspy.Tool] -> tool_calls: dspy.ToolCalls")
    student.set_lm(lm)
    tool = dspy.Tool(lambda query: query, name="lookup")
    example = dspy.Example(question="find this", tools=[tool]).with_inputs("question", "tools")
    with dspy.context(adapter=adapter_type(use_native_function_calling=True, use_json_adapter_fallback=False)):
        dspy.BootstrapFinetune(num_threads=1).compile(student, trainset=[example])
    target = provider.data[0]["messages"][-1]["content"]
    assert "lookup" in target and "tool-query-91" in target


# Define a simple metric function for testing
def simple_metric(example, prediction, trace=None):
    return example.output == prediction.output


examples = [
    Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    Example(input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs("input"),
]
trainset = [examples[0]]


class TraceIdentityAdapter(dspy.ChatAdapter):
    def format_finetune_data(self, signature, demos, inputs, outputs):
        return {"inputs": inputs, "outputs": outputs}


def make_two_predictor_trace():
    return [
        {
            "trace": [
                (Predict("input -> output"), {"input": "predictor-0"}, {"output": "zero"}),
                (Predict("input -> output"), {"input": "predictor-1"}, {"output": "one"}),
            ]
        }
    ]


def test_bootstrap_finetune_initialization():
    """Test BootstrapFinetune initialization with various parameters."""
    bootstrap = BootstrapFinetune(metric=simple_metric)
    assert bootstrap.metric == simple_metric, "Metric not correctly initialized"
    assert bootstrap.multitask == True, "Multitask should default to True"


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def test_compile_with_predict_instances():
    """Test BootstrapFinetune compilation with Predict instances."""
    # Create SimpleModule instances for student and teacher
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")

    lm = DummyLM([{"output": "blue"}, {"output": "Ring-ding-ding-ding-dingeringeding!"}])
    dspy.configure(lm=lm)

    # Set LM for both student and teacher
    student.set_lm(lm)
    teacher.set_lm(lm)

    bootstrap = BootstrapFinetune(metric=simple_metric)

    # Mock the fine-tuning process since DummyLM doesn't support it
    with patch.object(bootstrap, "finetune_lms") as mock_finetune:
        mock_finetune.return_value = {(lm, None): lm}
        compiled_student = bootstrap.compile(student, teacher=teacher, trainset=trainset)

        assert compiled_student is not None, "Failed to compile student"
        assert hasattr(compiled_student, "_compiled") and compiled_student._compiled, "Student compilation flag not set"

        mock_finetune.assert_called_once()


def test_error_handling_missing_lm():
    """Test error handling when predictor doesn't have an LM assigned."""

    lm = DummyLM([{"output": "test"}])
    dspy.configure(lm=lm)

    student = SimpleModule("input -> output")
    # Intentionally NOT setting LM for the student module

    bootstrap = BootstrapFinetune(metric=simple_metric)

    # This should raise ValueError about missing LM and hint to use set_lm
    try:
        bootstrap.compile(student, trainset=trainset)
        assert False, "Should have raised ValueError for missing LM"
    except ValueError as e:
        assert "does not have an LM assigned" in str(e)
        assert "set_lm" in str(e)


def test_prepare_finetune_data_filters_to_requested_predictor():
    bootstrap = BootstrapFinetune(adapter=TraceIdentityAdapter(), exclude_demos=True)

    data, _ = bootstrap._prepare_finetune_data(
        trace_data=make_two_predictor_trace(),
        lm=DummyLM([]),
        pred_ind=1,
    )

    assert data == [{"inputs": {"input": "predictor-1"}, "outputs": {"output": "one"}}]


def test_prepare_finetune_data_includes_all_predictors_without_filter():
    bootstrap = BootstrapFinetune(adapter=TraceIdentityAdapter(), exclude_demos=True)

    data, _ = bootstrap._prepare_finetune_data(
        trace_data=make_two_predictor_trace(),
        lm=DummyLM([]),
        pred_ind=None,
    )

    assert sorted(data, key=lambda item: item["inputs"]["input"]) == [
        {"inputs": {"input": "predictor-0"}, "outputs": {"output": "zero"}},
        {"inputs": {"input": "predictor-1"}, "outputs": {"output": "one"}},
    ]
