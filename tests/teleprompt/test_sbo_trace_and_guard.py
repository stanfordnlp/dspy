import dspy
from dspy import Example, Prediction
from dspy.adapters.chat_adapter import ChatAdapter, FieldInfoWithName
from dspy.signatures.field import OutputField
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import AdapterParseError

from dspy.teleprompt.sbo import BundleEntry, SBOProgramCallError, SemanticBundleOptimization, SemanticBundleOptimizationLite


class SBOTestLM(DummyLM):
    """Dummy LM that returns structured outputs based on which SBO signature is calling."""

    def __init__(self, judge_score: str = "1.0"):
        super().__init__([], adapter=ChatAdapter())
        self.judge_score = judge_score

    def _format_fields(self, field_names_and_values: dict[str, object]) -> str:
        fields_with_values = {
            FieldInfoWithName(name=field_name, info=OutputField()): value
            for field_name, value in field_names_and_values.items()
        }
        return self.adapter.format_field_with_value(fields_with_values)

    def __call__(self, prompt=None, messages=None, **kwargs):
        text = str(messages or prompt or "")
        if "verification_json" in text or "LiteConstraintVerifier" in text:
            payload = (
                '{"active":[{"id":"A1","label":"resolved"}],'
                '"watchlist":[],"blocking_regression":false,"summary":"ok"}'
            )
            outputs = [self._format_fields({"verification_json": payload})]
        elif "candidate_prompt" in text and "reference_prompt" in text:
            outputs = [self._format_fields({"score": self.judge_score})]
        elif "num_candidates" in text and "active_critique_bundle" in text:
            outputs = [self._format_fields({"candidates": ["improved instruction"]})]
        elif "failure_feedback" in text or "failure_context" in text:
            outputs = [self._format_fields({"critique": "make the instruction improved"})]
        elif "instruction_template" in text:
            outputs = [self._format_fields({"critique": "make the instruction improved"})]
        else:
            outputs = [self._format_fields({"answer": "ok"})]

        entry = {
            "prompt": prompt,
            "messages": messages,
            "kwargs": {**self.kwargs, **kwargs},
            "outputs": outputs,
            "usage": 0,
            "cost": 0,
        }
        self.update_history(entry)
        return outputs

    def copy(self, **kwargs):
        new_lm = SBOTestLM(self.judge_score)
        new_lm.kwargs = {**self.kwargs, **kwargs}
        new_lm.cache = kwargs.get("cache", self.cache)
        return new_lm


def _make_sbo_test_lm(judge_score: str = "1.0") -> SBOTestLM:
    return SBOTestLM(judge_score)


class ToyProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("context, question -> answer")

    def forward(self, context, question):
        instruction = self.predictor.signature.instructions or ""
        return Prediction(answer="yes" if "improved" in instruction and context else "no")


class FakeHistoryLM(SBOTestLM):
    def __call__(self, prompt=None, messages=None, **kwargs):
        if prompt and not messages:
            outputs = ["not parseable"]
            entry = {
                "prompt": prompt,
                "messages": [{"role": "user", "content": prompt}],
                "kwargs": {**self.kwargs, **kwargs},
                "outputs": outputs,
                "usage": 0,
                "cost": 0,
            }
            self.update_history(entry)
            return outputs
        return super().__call__(prompt=prompt, messages=messages, **kwargs)

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
    lm = _make_sbo_test_lm(judge_score)
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


def test_sbo_loss_first_accepts_improvement_with_non_positive_predicted_improvement():
    result = _compile_with_judge_score("-1.0", num_judge_samples=1, num_eval_samples=1)

    iteration = result.trace["iterations"][0]
    assert iteration["predicted_improvement"] <= 0
    assert iteration["positive_predicted_improvement"] == 0.0
    assert iteration["actual_improvement"] > 0
    assert iteration["step_type"] == "serious"
    assert result.num_serious_steps == 1
    assert result.num_null_steps == 0


def test_sbo_parser_ignores_negative_candidate_marker():
    optimizer = SemanticBundleOptimization(metric=metric)
    response = """CANDIDATE -3:
This is commentary, not a valid candidate marker.

CANDIDATE 1:
valid instruction"""

    texts, warnings = optimizer._parse_candidate_response(response)

    assert texts == ["valid instruction"]
    assert "ignored_negative_candidate_marker" in warnings


def test_sbo_exact_null_self_cut_enforces_candidate_loss():
    lm = _make_sbo_test_lm("-1.0")
    optimizer = SemanticBundleOptimization(metric=metric, num_judge_samples=1)
    prompts = {"predictor": "candidate"}
    entry = BundleEntry(
        prompt=prompts,
        loss=0.4,
        critique="failed to improve",
        iteration=1,
        self_score=0.0,
        lambda_value=1.0,
        kind="null_self_cut",
        exact_cut_signature=optimizer._prompt_signature(prompts),
        exact_cut_loss=0.4,
    )

    value = optimizer._compute_model_value(prompts, [(0, entry)], 1.0, lm)

    assert value >= 0.4


def test_sbo_lite_accepts_improving_candidate_without_blocking_regression():
    lm = _make_sbo_test_lm("1.0")
    dspy.configure(lm=lm)
    trainset = [Example(context="c", question="q", answer="yes").with_inputs("context", "question")]
    valset = [Example(context="c", question="q", answer="yes").with_inputs("context", "question")]
    optimizer = SemanticBundleOptimizationLite(
        metric=metric,
        judge_lm=lm,
        proposer_lm=lm,
        critic_lm=lm,
        num_candidates=1,
        max_iterations=1,
        judge_temperature=0.7,
    )

    optimizer.compile(ToyProgram(), trainset=trainset, valset=valset)
    result = optimizer.result
    iteration = result.trace["iterations"][0]

    assert result.num_serious_steps == 1
    assert result.trace["algorithm"] == "sbo_lite"
    assert iteration["candidates"][0]["verifier"]["blocking_regression"] is False
    assert iteration["step_type"] == "serious"


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
