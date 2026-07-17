"""Tests for AvatarOptimizer.thread_safe_evaluator.

These tests isolate the parallel-evaluation path from AvatarOptimizer.__init__ (which
currently constructs a dspy.TypedPredictor and is affected by the unrelated #7931). We
build the optimizer via ``__new__`` and set ``self.metric`` directly, since
``thread_safe_evaluator`` only depends on ``self.metric`` plus dspy settings.
"""

import threading

import pytest

import dspy
from dspy.teleprompt.avatar_optimizer import AvatarOptimizer
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback
from dspy.utils.dummies import DummyLM


def _make_optimizer(metric):
    optimizer = AvatarOptimizer.__new__(AvatarOptimizer)
    optimizer.metric = metric
    return optimizer


def _devset(n):
    return [dspy.Example(question=f"q{i}", answer="ok").with_inputs("question") for i in range(n)]


class ContextReadingActor(dspy.Module):
    """Actor that requires an LM to be visible in the thread that runs it.

    This mirrors the real failure in issue #10053: the actor reads ``dspy.settings.lm``,
    which is only supplied via a ``dspy.context`` override. If that override does not reach
    the worker thread, the actor raises and the example scores 0.
    """

    def forward(self, **kwargs):
        if dspy.settings.lm is None:
            raise ValueError("no LM configured in this thread")
        return dspy.Prediction(answer="ok")


class FailingActor(dspy.Module):
    """Actor that raises for a specific input and succeeds otherwise."""

    def forward(self, question=None, **kwargs):
        if question == "boom":
            raise RuntimeError("boom")
        return dspy.Prediction(answer="ok")


class AlwaysFailingActor(dspy.Module):
    """Actor that raises for every input."""

    def forward(self, **kwargs):
        raise RuntimeError("boom")


def _match_metric(example, prediction):
    return 1.0 if getattr(prediction, "answer", None) == "ok" else 0.0


def test_context_override_propagates_to_worker_threads():
    """Regression for #10053: a dspy.context override must reach worker threads.

    Previously thread_safe_evaluator used a bare ThreadPoolExecutor, so the
    ``dspy.context(lm=...)`` override never reached workers, the actor raised, and the
    average score collapsed to 0.0. Routing through ParallelExecutor re-applies the
    override in each worker, so the score is now correct (1.0).
    """
    optimizer = _make_optimizer(_match_metric)
    devset = _devset(6)
    actor = ContextReadingActor()

    # lm is provided ONLY via the context override (never globally configured), so the
    # override must transit into the worker threads for the actor to succeed.
    with dspy.context(lm=DummyLM([{"answer": "ok"}])):
        avg, results = optimizer.thread_safe_evaluator(devset, actor, return_outputs=True, num_threads=4)

    assert avg == 1.0
    assert len(results) == len(devset)
    assert all(score == 1.0 for _, _, score in results)


def test_results_preserve_devset_order():
    """Results must be returned in devset order regardless of worker completion order."""
    optimizer = _make_optimizer(_match_metric)
    devset = _devset(12)
    actor = ContextReadingActor()

    with dspy.context(lm=DummyLM([{"answer": "ok"}])):
        _, results = optimizer.thread_safe_evaluator(devset, actor, return_outputs=True, num_threads=4)

    assert [example for example, _, _ in results] == devset


def test_failing_example_scores_zero_and_counts_toward_average():
    """An example whose actor raises is recorded as (example, None, 0) and still counted."""
    optimizer = _make_optimizer(_match_metric)
    devset = [
        dspy.Example(question="ok-1", answer="ok").with_inputs("question"),
        dspy.Example(question="boom", answer="ok").with_inputs("question"),
        dspy.Example(question="ok-2", answer="ok").with_inputs("question"),
        dspy.Example(question="ok-3", answer="ok").with_inputs("question"),
    ]
    actor = FailingActor()

    with dspy.context(lm=DummyLM([{"answer": "ok"}])):
        avg, results = optimizer.thread_safe_evaluator(devset, actor, return_outputs=True, num_threads=4)

    # 3 successes out of 4 examples: the failure counts toward the denominator.
    assert avg == pytest.approx(3.0 / 4.0)

    failed = [(example, prediction, score) for example, prediction, score in results if example.question == "boom"]
    assert len(failed) == 1
    _, prediction, score = failed[0]
    assert prediction is None
    assert score == 0


def test_evaluation_tolerates_more_failures_than_default_max_errors():
    """Every failing example scores 0 without aborting, even past dspy.settings.max_errors.

    The legacy evaluator swallowed every per-example exception, so an unlimited number of
    failures never cancelled the run. Regression guard: a devset with far more failures than
    the default max_errors (10) must still complete with an all-zero average instead of
    raising "Execution cancelled due to errors or interruption.".
    """
    optimizer = _make_optimizer(_match_metric)
    n = dspy.settings.max_errors + 5
    devset = _devset(n)
    actor = AlwaysFailingActor()

    with dspy.context(lm=DummyLM([{"answer": "ok"}])):
        avg, results = optimizer.thread_safe_evaluator(devset, actor, return_outputs=True, num_threads=4)

    assert avg == 0.0
    assert len(results) == n
    assert all(prediction is None and score == 0 for _, prediction, score in results)


def test_process_example_is_the_per_example_override_point():
    """A subclass overriding process_example must still drive every example.

    process_example is public API, so the parallel-evaluation path has to route through it
    rather than inlining the work; otherwise an override is silently ignored. This pins the
    existing signature -- ``process_example(self, actor, example, return_outputs)`` -- so a
    subclass written against the pre-change API keeps working. That matters because an arity
    mismatch would raise inside a worker, where the error is swallowed and surfaces only as a
    silently zeroed score.
    """

    class CountingOptimizer(AvatarOptimizer):
        def __init__(self, metric):
            self.metric = metric
            self.calls = []
            self.lock = threading.Lock()

        def process_example(self, actor, example, return_outputs):
            with self.lock:
                self.calls.append(example.question)
            result = super().process_example(actor, example, return_outputs)
            # Override the score to prove the subclass's return value is what gets used.
            if return_outputs:
                example, prediction, score = result
                return example, prediction, score * 2
            return result * 2

    optimizer = CountingOptimizer(_match_metric)
    devset = _devset(4)
    actor = ContextReadingActor()

    with dspy.context(lm=DummyLM([{"answer": "ok"}])):
        avg, _ = optimizer.thread_safe_evaluator(devset, actor, return_outputs=True, num_threads=4)

    assert sorted(optimizer.calls) == sorted(example.question for example in devset)
    assert avg == 2.0


def test_callback_handlers_fire_inside_worker_threads():
    """Callback handlers registered via settings must fire in the worker threads.

    This is the #10043 parity point: the ``callbacks`` setting now transits into workers via
    ParallelExecutor's thread-local override re-application, so per-example actor calls invoke
    the handlers. (ACTIVE_CALL_ID call-ancestry transport is covered separately below.)
    """

    class RecordingCallback(BaseCallback):
        def __init__(self):
            self.start_threads = []
            self.lock = threading.Lock()

        def on_module_start(self, call_id, instance, inputs):
            with self.lock:
                self.start_threads.append(threading.get_ident())

    callback = RecordingCallback()
    optimizer = _make_optimizer(_match_metric)
    devset = _devset(6)
    actor = ContextReadingActor()

    with dspy.context(lm=DummyLM([{"answer": "ok"}]), callbacks=[callback]):
        optimizer.thread_safe_evaluator(devset, actor, return_outputs=True, num_threads=4)

    # One on_module_start per example, and at least one fired off the main thread.
    assert len(callback.start_threads) == len(devset)
    assert any(tid != threading.main_thread().ident for tid in callback.start_threads)


def test_active_call_id_ancestry_propagates_to_workers():
    """The parent ACTIVE_CALL_ID should be visible to callbacks running in worker threads."""

    class ParentRecordingCallback(BaseCallback):
        def __init__(self):
            self.parent_call_ids = []
            self.lock = threading.Lock()

        def on_module_start(self, call_id, instance, inputs):
            with self.lock:
                self.parent_call_ids.append(ACTIVE_CALL_ID.get())

    callback = ParentRecordingCallback()
    optimizer = _make_optimizer(_match_metric)
    devset = _devset(4)
    actor = ContextReadingActor()

    parent_id = "parent-call-id"
    token = ACTIVE_CALL_ID.set(parent_id)
    try:
        with dspy.context(lm=DummyLM([{"answer": "ok"}]), callbacks=[callback]):
            optimizer.thread_safe_evaluator(devset, actor, return_outputs=True, num_threads=4)
    finally:
        ACTIVE_CALL_ID.reset(token)

    assert callback.parent_call_ids
    assert all(pid == parent_id for pid in callback.parent_call_ids)
