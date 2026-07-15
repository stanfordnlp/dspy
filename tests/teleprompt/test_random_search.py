import asyncio

import pytest

import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.events import (
    CandidateProposed,
    CandidateSelected,
    RandomSearchCandidateProposed,
)
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback, _emit_optimizer_event
from dspy.utils.dummies import DummyLM


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)


def simple_metric(example, prediction, trace=None):
    return example.output == prediction.output


class IdentityModule(dspy.Module):
    def forward(self, input):
        return dspy.Prediction(output=input)


def identity_metric(example, prediction, trace=None):
    return float(example.output == prediction.output)


class RandomSearchCallback(BaseCallback):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.optimizer_start = None
        self.optimizer_end = None
        self.events = []
        self.evaluations = {}
        self.evaluation_order = []

    def on_optimizer_start(self, call_id, instance, inputs):
        if instance is self.optimizer:
            self.optimizer_start = {"call_id": call_id, "parent_call_id": ACTIVE_CALL_ID.get(), "inputs": inputs}

    def on_optimizer_event(self, call_id, instance, event):
        if instance is self.optimizer:
            self.events.append((call_id, event))

    def on_optimizer_end(self, call_id, outputs, exception):
        if self.optimizer_start is not None and call_id == self.optimizer_start["call_id"]:
            self.optimizer_end = {"call_id": call_id, "outputs": outputs, "exception": exception}

    def on_evaluate_start(self, call_id, instance, inputs):
        self.evaluation_order.append(call_id)
        self.evaluations[call_id] = {
            "parent_call_id": ACTIVE_CALL_ID.get(),
            "inputs": inputs,
            "outputs": None,
            "exception": None,
        }

    def on_evaluate_end(self, call_id, outputs, exception):
        self.evaluations[call_id]["outputs"] = outputs
        self.evaluations[call_id]["exception"] = exception


def make_identity_trainset():
    return [
        Example(input="correct", output="correct").with_inputs("input"),
        Example(input="incorrect", output="different").with_inputs("input"),
    ]


def test_basic_workflow():
    """Test to ensure the basic compile flow runs without errors."""
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")

    lm = DummyLM(
        [
            "Initial thoughts",
            "Finish[blue]",  # Expected output for both training and validation
        ]
    )
    dspy.configure(lm=lm)

    optimizer = BootstrapFewShotWithRandomSearch(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    trainset = [
        Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
        Example(input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs("input"),
    ]
    optimizer.compile(student, teacher=teacher, trainset=trainset)


def test_restrict_matching_no_candidate_seed_raises_clear_error():
    """restrict that matches no candidate seed should raise ValueError, not UnboundLocalError."""
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")

    lm = DummyLM(["Initial thoughts", "Finish[blue]"])
    dspy.configure(lm=lm)

    optimizer = BootstrapFewShotWithRandomSearch(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    trainset = [
        Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    ]

    with pytest.raises(ValueError, match="restrict"):
        optimizer.compile(student, teacher=teacher, trainset=trainset, restrict=[999])


def test_restrict_as_single_use_iterator_still_matches_a_valid_seed():
    """A single-use iterable `restrict` must not be exhausted by the upfront validation check."""
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")

    lm = DummyLM(["Initial thoughts", "Finish[blue]"])
    dspy.configure(lm=lm)

    optimizer = BootstrapFewShotWithRandomSearch(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    trainset = [
        Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
    ]

    # -3 (zero-shot) is a valid seed; a plain iterator is single-use.
    result = optimizer.compile(student, teacher=teacher, trainset=trainset, restrict=iter([-3]))
    assert result is not None


def test_random_search_correlates_candidates_with_evaluations_and_selection():
    optimizer = BootstrapFewShotWithRandomSearch(metric=identity_metric, num_threads=2)
    callback = RandomSearchCallback(optimizer)

    with dspy.context(callbacks=[callback]):
        result = optimizer.compile(IdentityModule(), trainset=make_identity_trainset(), restrict=[-3, -1])

    run_id = callback.optimizer_start["call_id"]
    assert callback.optimizer_start["parent_call_id"] is None
    assert callback.optimizer_end == {"call_id": run_id, "outputs": result, "exception": None}
    assert all(call_id == run_id for call_id, _ in callback.events)

    events = [event for _, event in callback.events]
    assert events == [
        RandomSearchCandidateProposed(candidate_index=0, seed=-3, kind="zero_shot"),
        RandomSearchCandidateProposed(candidate_index=1, seed=-1, kind="unshuffled_bootstrap"),
        CandidateSelected(candidate_index=0, score=50.0),
    ]
    proposed = events[:2]
    assert all(isinstance(event, CandidateProposed) for event in proposed)
    assert all(not hasattr(event, "program") for event in events)

    assert len(callback.evaluations) == 2
    for candidate_index, call_id in enumerate(callback.evaluation_order):
        evaluation = callback.evaluations[call_id]
        assert evaluation["parent_call_id"] == run_id
        assert evaluation["inputs"]["callback_metadata"] == {
            "metric_key": "eval_full",
            "candidate_index": candidate_index,
        }
        assert evaluation["outputs"].score == 50.0
        assert evaluation["exception"] is None

    selected = events[-1]
    selected_evaluation = callback.evaluations[callback.evaluation_order[selected.candidate_index]]
    assert selected.score == selected_evaluation["outputs"].score

    assert [candidate["seed"] for candidate in result.candidate_programs] == [-3, -1]
    assert all(set(candidate) == {"score", "subscores", "seed", "program"} for candidate in result.candidate_programs)


def test_random_search_selection_event_matches_early_stop_winner():
    metric_scores = iter([0.0, 1.0])

    def improving_metric(example, prediction, trace=None):
        return next(metric_scores)

    optimizer = BootstrapFewShotWithRandomSearch(metric=improving_metric, num_threads=1, stop_at_score=100.0)
    callback = RandomSearchCallback(optimizer)
    trainset = [Example(input="value", output="value").with_inputs("input")]

    with dspy.context(callbacks=[callback]):
        result = optimizer.compile(IdentityModule(), trainset=trainset, restrict=[-3, -2, -1])

    events = [event for _, event in callback.events]
    assert events == [
        RandomSearchCandidateProposed(candidate_index=0, seed=-3, kind="zero_shot"),
        RandomSearchCandidateProposed(candidate_index=1, seed=-2, kind="labeled_few_shot"),
        CandidateSelected(candidate_index=1, score=100.0),
    ]
    assert len(callback.evaluations) == 2
    assert [candidate["seed"] for candidate in result.candidate_programs] == [-2, -3]


def test_random_search_event_callback_failure_does_not_change_result():
    class FailingCallback(BaseCallback):
        def on_optimizer_event(self, call_id, instance, event):
            raise ValueError("tracker failed")

    optimizer = BootstrapFewShotWithRandomSearch(metric=identity_metric)

    with dspy.context(callbacks=[FailingCallback()]):
        result = optimizer.compile(IdentityModule(), trainset=make_identity_trainset(), restrict=[-3])

    assert result.candidate_programs[0]["seed"] == -3


@pytest.mark.asyncio
async def test_optimizer_event_rejects_completed_detached_context():
    class EventCallback(BaseCallback):
        def on_optimizer_event(self, call_id, instance, event):
            pytest.fail("A completed optimizer call must not emit an event")

    class AsyncOptimizer(dspy.Teleprompter):
        async def compile(self, student, *, trainset):
            async def emit_after_compile():
                await self.release.wait()
                _emit_optimizer_event(self, CandidateSelected(candidate_index=0, score=100.0))

            self.release = asyncio.Event()
            self.detached_task = asyncio.create_task(emit_after_compile())
            return student

    optimizer = AsyncOptimizer()

    with dspy.context(callbacks=[EventCallback()]):
        student = object()
        assert await optimizer.compile(student, trainset=[]) is student
        optimizer.release.set()
        with pytest.raises(RuntimeError, match="active optimizer compile"):
            await optimizer.detached_task
