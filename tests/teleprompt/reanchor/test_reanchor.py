"""ReAnchor on a stubbed System One client and a stubbed generative LM."""

import copy
import json
import logging

import pytest

import dspy
from dspy.experimental import ReAnchor
from tests.teleprompt.reanchor.fakes import ComputedLM, noul

source = ReAnchor.source


def leaning(state, name, q):
    """Jev leans high, 0.9 on a same pair and 0.7 on a different one, and reads a tricky same pair
    as different, so no threshold gets every pair right. The best thresholds lie between 0.7 and 0.9; ReAnchor picks the midpoint, 0.8."""
    pair = state["inputs"]["pair"]
    if pair.endswith("same-tricky"):
        return noul(0.7)
    return noul(0.9 if "same" in pair else 0.7)


class Sig(dspy.Signature):
    pair: str = dspy.InputField()
    match: bool = dspy.OutputField(desc="Are the two the same?")


@pytest.fixture(autouse=True)
def jev(system_one):
    return system_one(leaning)


@pytest.fixture(autouse=True)
def settings():
    # Configured rather than set with `dspy.context`, which would pin the client a test installs later.
    dspy.configure(adapter=dspy.JSONAdapter())


def examples(prefix=""):
    kinds = ["same"] * 6 + ["same-tricky"] * 4 + ["different"] * 8
    return [dspy.Example(pair=prefix + k, match=k.startswith("same")).with_inputs("pair") for k in kinds]


def metric(gold, pred, trace=None):
    return float(pred.match == gold.match)


def test_compile_fits_the_threshold_and_reports_the_training_scores():
    optimizer = ReAnchor(metric, num_threads=2)
    program = optimizer.compile(dspy.Predict(Sig), trainset=examples())
    assert isinstance(program, dspy.Predict)
    assert program.fields["match"]["threshold"] == 0.8
    assert optimizer.report["train_score_before"] == 0.5556 and optimizer.report["train_score"] == 0.7778
    assert optimizer.report["fitted"][0]["parameter"] == "threshold"
    assert "val_score" not in optimizer.report


def test_the_validation_set_is_scored_and_never_fitted_on():
    optimizer = ReAnchor(metric, num_threads=2)
    val = examples("v-")[:10]  # same pairs only, where the default threshold is already right
    program = optimizer.compile(dspy.Predict(Sig), trainset=examples(), valset=val)
    assert program.fields["match"]["threshold"] == 0.8
    assert optimizer.report["val_score_before"] == 1.0 and optimizer.report["val_score"] == 0.6


def test_compile_leaves_the_student_unchanged_and_marks_the_program_compiled():
    student = dspy.Predict(Sig)
    before = copy.deepcopy(student.fields)
    program = ReAnchor(metric, num_threads=2).compile(student, trainset=examples())
    assert student.fields == before
    assert program._compiled and program is not student


def test_a_metric_returning_a_prediction_is_read_by_its_score():
    graded = lambda gold, pred, trace=None: dspy.Prediction(score=metric(gold, pred), feedback="")  # noqa: E731
    program = ReAnchor(graded, num_threads=2).compile(dspy.Predict(Sig), trainset=examples())
    assert program.fields["match"]["threshold"] == 0.8


def test_a_module_holding_predictors_gets_each_one_calibrated():
    class Wrapper(dspy.Module):
        def __init__(self):
            super().__init__()
            self.judge = dspy.Predict(Sig)
            self.judges = [dspy.Predict(Sig)]

        def forward(self, pair):
            return self.judges[0](pair=pair) if self.judge(pair=pair).match else dspy.Prediction(match=False)

    program = ReAnchor(metric, num_threads=2).compile(Wrapper(), trainset=examples())
    assert isinstance(program, Wrapper)
    assert program.judge.fields["match"]["threshold"] == 0.8
    assert "threshold" in program.judges[0].fields["match"]


class Items(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = dspy.Predict(Sig)

    def forward(self, pairs):
        return dspy.Prediction(matches=[self.judge(pair=p).match for p in pairs])


def test_a_predictor_called_once_per_item_is_calibrated_from_the_program_metric():
    batch = ["same", "different", "different"]
    trainset = [dspy.Example(pairs=batch, matches=[k.startswith("same") for k in batch]).with_inputs("pairs")] * 4

    def items_metric(gold, pred, trace=None):
        return sum(a == b for a, b in zip(pred.matches, gold.matches, strict=True)) / len(gold.matches)

    program = ReAnchor(items_metric, num_threads=2).compile(Items(), trainset=trainset)
    assert isinstance(program, Items) and program.judge.fields["match"]["threshold"] == 0.8


def test_the_program_keeps_a_bound_client_and_callbacks(system_one):
    client = system_one(leaning)
    dspy.configure(lm=None)
    student = dspy.Predict(Sig, lm=client, callbacks=[])
    program = ReAnchor(metric, num_threads=2).compile(student, trainset=examples())
    assert program.lm is not None and program(pair="same").match is True


def test_source_writes_the_signature_and_the_fitted_parameters():
    program = ReAnchor(metric, num_threads=2).compile(dspy.Predict(Sig), trainset=examples())
    text = source(program)
    assert text.startswith("class Sig(dspy.Signature):")
    assert "program.fields = {'match': {'threshold': 0.8}}" in text


def test_log_dir_holds_the_report_and_source(tmp_path):
    ReAnchor(metric, num_threads=2, log_dir=tmp_path).compile(dspy.Predict(Sig), trainset=examples())
    assert json.loads((tmp_path / "report.json").read_text())["train_score"] == 0.7778
    assert "'threshold': 0.8" in (tmp_path / "source.py").read_text()


def test_compile_logs_each_stage():
    lines = []
    handler = logging.Handler()
    handler.emit = lambda record: lines.append(record.getMessage())
    log = logging.getLogger("dspy.teleprompt.reanchor.reanchor")
    log.addHandler(handler)
    try:
        ReAnchor(metric, num_threads=2).compile(dspy.Predict(Sig), trainset=examples(), valset=examples("v-"))
    finally:
        log.removeHandler(handler)
    assert lines == [
        "answering 18 training examples",
        "fitting thresholds, cuts, and weights",
        "calibrated: train 0.5556 -> 0.7778",
        "validation: 0.5556 -> 0.7778",
    ]


def test_empty_trainset_fails():
    with pytest.raises(ValueError, match="trainset"):
        ReAnchor(metric).compile(dspy.Predict(Sig), trainset=[])


def test_a_student_without_a_decision_output_fails():
    with pytest.raises(ValueError, match="decision output"):
        ReAnchor(metric).compile(dspy.Predict("pair -> answer: str"), trainset=examples())


def test_an_uncached_client_fails_unless_the_cache_is_waived(system_one):
    system_one(leaning, cache=False)
    with pytest.raises(ValueError, match="require_cache=False"):
        ReAnchor(metric).compile(dspy.Predict(Sig), trainset=examples())
    program = ReAnchor(metric, num_threads=2, require_cache=False).compile(dspy.Predict(Sig), trainset=examples())
    assert program.fields["match"]["threshold"] == 0.8


def test_a_predictor_config_that_turns_the_cache_off_fails(system_one):
    with pytest.raises(ValueError, match="require_cache=False"):
        ReAnchor(metric).compile(dspy.Predict(Sig, cache=False), trainset=examples())


def generative(inputs, evidence):
    """The LM answers True on every pair when asked for a bool. Asked for probabilities, it leans
    the same way as the System One client."""
    if not evidence:
        return {"match": True}
    pair = inputs["pair"]
    return {"match": noul(0.7 if pair.endswith("same-tricky") else 0.9 if "same" in pair else 0.7)}


def test_a_native_bool_on_a_generative_lm_is_promoted_when_probabilities_score_better():
    dspy.configure(lm=ComputedLM(generative, adapter=dspy.JSONAdapter()))
    student = dspy.Predict(Sig)
    assert student.fields == {}
    optimizer = ReAnchor(metric, num_threads=2, require_cache=False)
    program = optimizer.compile(student, trainset=examples())
    assert student.fields == {}
    assert program.fields == {"match": {"threshold": 0.8}}
    row = optimizer.report["fitted"][0]
    assert row["promoted"] is True and row["train_score_native"] == 0.5556 and row["train_score"] == 0.7778
    assert "program.fields = {'match': {'threshold': 0.8}}" in source(program)


def test_a_native_bool_stays_native_when_probabilities_do_not_score_better():
    def right(inputs, evidence):
        answer = inputs["pair"].startswith("same")
        return {"match": noul(0.9 if answer else 0.1)} if evidence else {"match": answer}

    dspy.configure(lm=ComputedLM(right, adapter=dspy.JSONAdapter()))
    optimizer = ReAnchor(metric, num_threads=2, require_cache=False)
    program = optimizer.compile(dspy.Predict(Sig), trainset=examples())
    assert program.fields == {}
    assert optimizer.report["fitted"][0]["skipped"] == "probabilities did not beat the native output"


def test_a_native_bool_stays_native_when_probabilities_win_only_one_example():
    pairs = [f"same{i}" for i in range(20)] + [f"different{i}" for i in range(20)]

    def one_miss(inputs, evidence):
        answer = inputs["pair"].startswith("same")
        if evidence:
            return {"match": noul(0.9 if answer else 0.1)}
        return {"match": answer and inputs["pair"] != "same0"}

    dspy.configure(lm=ComputedLM(one_miss, adapter=dspy.JSONAdapter()))
    train = [dspy.Example(pair=p, match=p.startswith("same")).with_inputs("pair") for p in pairs]
    optimizer = ReAnchor(metric, num_threads=2, require_cache=False)
    program = optimizer.compile(dspy.Predict(Sig), trainset=train)
    assert program.fields == {}
    assert optimizer.report["fitted"][0]["skipped"] == "probabilities did not beat the native output"


def test_a_partial_field_entry_is_filled_from_the_type_defaults(system_one):
    student = dspy.Predict(Sig)
    student.set_criteria("match", {"true": "The same item.", "false": "Two different items."})
    program = ReAnchor(metric, num_threads=2).compile(student, trainset=examples())
    assert program.fields["match"] == {
        "criteria": {"true": "The same item.", "false": "Two different items."},
        "threshold": 0.8,
    }


def test_the_calibrated_program_saves_and_loads(tmp_path):
    program = ReAnchor(metric, num_threads=2).compile(dspy.Predict(Sig), trainset=examples())
    program.save(tmp_path / "program.json")
    restored = dspy.Predict(Sig)
    restored.load(tmp_path / "program.json")
    assert restored.fields == {"match": {"threshold": 0.8}}
