import dspy
from dspy import Example, Prediction
from dspy.utils.exceptions import AdapterParseError

from dspy.teleprompt.sbo import SBOProgramCallError, SemanticBundleOptimization


class FakeStringLM:
    model = "fake/test"
    cache = False
    kwargs = {"temperature": 0.7}

    def __init__(self, judge_score: str = "1.0"):
        self.judge_score = judge_score

    def __call__(self, prompt=None, **kwargs):
        text = prompt or ""
        if "Output ONLY a single number" in text:
            return [self.judge_score]
        if "CANDIDATE" in text and "Candidates:" in text:
            return ["CANDIDATE 1:\nimproved instruction"]
        if "Critique:" in text:
            return ["make the instruction improved"]
        return ["ok"]

    def copy(self, **kwargs):
        new_lm = FakeStringLM(self.judge_score)
        new_lm.kwargs = {**self.kwargs, **kwargs}
        new_lm.cache = kwargs.get("cache", self.cache)
        return new_lm


class ToyProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("context, question -> answer")

    def forward(self, context, question):
        instruction = self.predictor.signature.instructions or ""
        return Prediction(answer="yes" if "improved" in instruction and context else "no")


class FakeHistoryLM(FakeStringLM):
    def __init__(self):
        super().__init__()
        self.history = []

    def __call__(self, prompt=None, **kwargs):
        self.history.append({
            "messages": [{"role": "user", "content": prompt}],
            "kwargs": kwargs,
            "outputs": ["not parseable"],
            "model": self.model,
        })
        return ["not parseable"]

    def copy(self, **kwargs):
        new_lm = FakeHistoryLM()
        new_lm.kwargs = {**self.kwargs, **kwargs}
        new_lm.cache = kwargs.get("cache", self.cache)
        return new_lm


class ErrorAfterLMProgram(dspy.Module):
    def forward(self, question):
        dspy.settings.lm(f"question: {question}")
        raise ValueError("parse failed")


class ParseThenSuccessProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.signature = dspy.Signature("question -> answer")

    def forward(self, question):
        dspy.settings.lm(f"question: {question}")
        self.calls += 1
        if self.calls == 1:
            raise AdapterParseError(
                adapter_name="JSONAdapter",
                signature=self.signature,
                lm_response="{bad}",
            )
        return Prediction(answer="ok")


def metric(example, pred, trace=None):
    return 1.0 if pred.answer == example.answer else 0.0


def _compile_with_judge_score(judge_score: str, **kwargs):
    lm = FakeStringLM(judge_score)
    dspy.configure(lm=lm)
    trainset = [Example(context="c", question="q", answer="yes").with_inputs("context", "question")]
    valset = [Example(context="c", question="q", answer="yes").with_inputs("context", "question")]
    optimizer = SemanticBundleOptimization(
        metric=metric,
        judge_lm=lm,
        proposer_lm=lm,
        critic_lm=lm,
        num_candidates=1,
        max_iterations=1,
        judge_temperature=0.7,
        **kwargs,
    )
    optimizer.compile(ToyProgram(), trainset=trainset, valset=valset)
    return optimizer.result


def test_sbo_records_detailed_trace_for_stochastic_samples():
    result = _compile_with_judge_score("1.0", num_judge_samples=2, num_eval_samples=2)

    assert result.num_serious_steps == 1
    trace = result.trace
    iteration = trace["iterations"][0]
    verifier_score_trace = iteration["verifier"]["candidates"][0]["bundle_scores"][0]["semantic_score_trace"]
    initial_critique = trace["initial"]["critique_generation"]

    assert iteration["proposer"]["proposer_prompt"]
    assert iteration["proposer"]["prompt_text"]
    assert initial_critique["instruction_prompt_text"]
    assert initial_critique["task_prompt_texts"]
    assert initial_critique["feedback_texts"]
    assert initial_critique["instruction_prompt_text"] == "Given the fields `context`, `question`, produce the fields `answer`."
    assert initial_critique["task_prompt_texts"][0].startswith("Given the fields `context`, `question`")
    assert "context: 'c'" in initial_critique["task_prompt_texts"][0]
    assert "question: 'q'" in initial_critique["task_prompt_texts"][0]
    assert "Expected Outputs" not in initial_critique["task_prompt_texts"][0]
    assert "Example 1" not in initial_critique["task_prompt_texts"][0]
    assert "Expected Outputs" in initial_critique["feedback_texts"][0]
    assert len(verifier_score_trace["samples"]) == 2
    assert len(trace["evaluations"][0]["examples"][0]["samples"]) == 2
    assert verifier_score_trace["samples"][0]["cache"] is False
    assert verifier_score_trace["samples"][0]["rollout_id"] is not None


def test_sbo_non_positive_predicted_improvement_is_null_step():
    result = _compile_with_judge_score("-1.0", num_judge_samples=1, num_eval_samples=1)

    iteration = result.trace["iterations"][0]
    assert iteration["predicted_improvement"] <= 0
    assert iteration["step_type"] == "null"
    assert iteration["null_reason"] == "non_positive_predicted_improvement"
    assert result.num_serious_steps == 0
    assert result.num_null_steps == 1


def test_program_call_error_preserves_actual_lm_history():
    lm = FakeHistoryLM()
    dspy.configure(lm=lm)
    optimizer = SemanticBundleOptimization(metric=metric)

    try:
        optimizer._run_program_with_lm_trace(ErrorAfterLMProgram(), {"question": "q"}, lm)
    except SBOProgramCallError as e:
        assert isinstance(e.original_error, ValueError)
        assert len(e.actual_task_lm_calls) == 1
        assert e.actual_task_lm_calls[0]["messages"][0]["content"] == "question: q"
        assert e.actual_task_lm_calls[0]["outputs"] == ["not parseable"]
    else:
        raise AssertionError("Expected SBOProgramCallError")


def test_program_call_retries_parse_failure_and_records_attempts():
    lm = FakeHistoryLM()
    dspy.configure(lm=lm)
    optimizer = SemanticBundleOptimization(metric=metric, parse_failure_retries=1)

    pred, calls = optimizer._run_program_with_lm_trace(ParseThenSuccessProgram(), {"question": "q"}, lm)

    assert pred.answer == "ok"
    assert len(calls) == 2
    assert calls[0]["attempt_idx"] == 0
    assert calls[0]["is_retry"] is False
    assert calls[1]["attempt_idx"] == 1
    assert calls[1]["is_retry"] is True
    assert calls[0]["messages"][0]["content"] == "question: q"
    assert calls[1]["messages"][0]["content"] == "question: q"
