from unittest.mock import Mock

from dspy.teleprompt import mipro_optimizer_v2
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2
from dspy.utils.dummies import DummyLM


def _make_optimizer(sample_metric, population_metric):
    return MIPROv2(
        metric=sample_metric,
        population_metric=population_metric,
        prompt_model=DummyLM({}),
        task_model=DummyLM({}),
        auto=None,
        num_candidates=1,
    )


def test_mipro_compile_wires_population_metric_into_evaluate(monkeypatch):
    def sample_metric(example, prediction):
        return True

    def population_metric(examples, predictions):
        return 1.0

    captured = {}

    class FakeEvaluate:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class Student:
        def deepcopy(self):
            return self

    optimizer = _make_optimizer(sample_metric, population_metric)
    monkeypatch.setattr(mipro_optimizer_v2, "Evaluate", FakeEvaluate)
    monkeypatch.setattr(optimizer, "_bootstrap_fewshot_examples", Mock(return_value=[]))
    monkeypatch.setattr(optimizer, "_propose_instructions", Mock(return_value={}))
    monkeypatch.setattr(optimizer, "_optimize_prompt_parameters", Mock(side_effect=lambda program, *args: program))

    optimizer.compile(
        Student(),
        trainset=[object()],
        valset=[object()],
        num_trials=1,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        minibatch=False,
        program_aware_proposer=False,
        data_aware_proposer=False,
        tip_aware_proposer=False,
        fewshot_aware_proposer=False,
    )

    assert captured["metric"] is sample_metric
    assert captured["population_metric"] is population_metric


def test_mipro_bootstrapping_uses_only_sample_metric(monkeypatch):
    def sample_metric(example, prediction):
        return True

    def population_metric(examples, predictions):
        return 1.0

    captured = {}

    def fake_create_n_fewshot_demo_sets(**kwargs):
        captured.update(kwargs)
        return []

    optimizer = _make_optimizer(sample_metric, population_metric)
    monkeypatch.setattr(mipro_optimizer_v2, "create_n_fewshot_demo_sets", fake_create_n_fewshot_demo_sets)

    optimizer._bootstrap_fewshot_examples(
        program=object(),
        trainset=[object()],
        seed=9,
        teacher=None,
        num_fewshot_candidates=1,
        max_bootstrapped_demos=1,
        max_labeled_demos=1,
        max_errors=1,
        metric_threshold=None,
    )

    assert captured["metric"] is sample_metric
    assert "population_metric" not in captured
