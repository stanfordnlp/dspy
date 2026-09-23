"""ReAnchor on a stubbed System One client."""

import copy
import json
import logging

import pytest

import dspy
from dspy.experimental import Decide, ReAnchor
from tests.teleprompt.reanchor.fakes import noul

source = ReAnchor.source


def leaning(state, name, q):
    """Jev leans high, 0.9 on a same pair and 0.7 on a different one, and reads a tricky same pair
    as different, so no threshold gets every pair right. The best threshold is 0.75."""
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
    program = optimizer.compile(Decide(Sig), trainset=examples())
    assert isinstance(program, Decide)
    assert program.fields["match"]["threshold"] == 0.75
    assert optimizer.report["train_score_before"] == 0.5556 and optimizer.report["train_score"] == 0.7778
    assert optimizer.report["fitted"][0]["parameter"] == "threshold"
    assert "val_score" not in optimizer.report


def test_the_validation_set_is_scored_and_never_fitted_on():
    optimizer = ReAnchor(metric, num_threads=2)
    val = examples("v-")[:10]  # same pairs only, where the default threshold is already right
    program = optimizer.compile(Decide(Sig), trainset=examples(), valset=val)
    assert program.fields["match"]["threshold"] == 0.75
    assert optimizer.report["val_score_before"] == 1.0 and optimizer.report["val_score"] == 0.6


def test_compile_leaves_the_student_unchanged_and_marks_the_program_compiled():
    student = Decide(Sig)
    before = copy.deepcopy(student.fields)
    program = ReAnchor(metric, num_threads=2).compile(student, trainset=examples())
    assert student.fields == before
    assert program._compiled and program is not student


def test_a_metric_returning_a_prediction_is_read_by_its_score():
    graded = lambda gold, pred, trace=None: dspy.Prediction(score=metric(gold, pred), feedback="")  # noqa: E731
    program = ReAnchor(graded, num_threads=2).compile(Decide(Sig), trainset=examples())
    assert program.fields["match"]["threshold"] == 0.75


def test_a_module_holding_decides_gets_each_one_calibrated():
    class Wrapper(dspy.Module):
        def __init__(self):
            super().__init__()
            self.judge = Decide(Sig)
            self.judges = [Decide(Sig)]

        def forward(self, pair):
            return self.judges[0](pair=pair) if self.judge(pair=pair).match else dspy.Prediction(match=False)

    program = ReAnchor(metric, num_threads=2).compile(Wrapper(), trainset=examples())
    assert isinstance(program, Wrapper)
    assert program.judge.fields["match"]["threshold"] == 0.75
    assert "threshold" in program.judges[0].fields["match"]


class Items(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = Decide(Sig)

    def forward(self, pairs):
        return dspy.Prediction(matches=[self.judge(pair=p).match for p in pairs])


def test_a_decide_called_once_per_item_is_calibrated_from_the_program_metric():
    batch = ["same", "different", "different"]
    trainset = [dspy.Example(pairs=batch, matches=[k.startswith("same") for k in batch]).with_inputs("pairs")] * 4

    def items_metric(gold, pred, trace=None):
        return sum(a == b for a, b in zip(pred.matches, gold.matches, strict=True)) / len(gold.matches)

    program = ReAnchor(items_metric, num_threads=2).compile(Items(), trainset=trainset)
    assert isinstance(program, Items) and program.judge.fields["match"]["threshold"] == 0.75


def test_the_program_keeps_the_client_and_callbacks(system_one):
    client = system_one(leaning)
    dspy.configure(system_one=None)
    student = Decide(Sig, client=client, callbacks=[])
    program = ReAnchor(metric, num_threads=2).compile(student, trainset=examples())
    assert program.client is not None and program(pair="same").match is True


def test_source_writes_the_signature_and_the_fitted_parameters():
    program = ReAnchor(metric, num_threads=2).compile(Decide(Sig), trainset=examples())
    text = source(program)
    assert text.startswith("class Sig(dspy.Signature):")
    assert "program.fields = {'match': {'threshold': 0.75}}" in text


def test_log_dir_holds_the_report_and_source(tmp_path):
    ReAnchor(metric, num_threads=2, log_dir=tmp_path).compile(Decide(Sig), trainset=examples())
    assert json.loads((tmp_path / "report.json").read_text())["train_score"] == 0.7778
    assert "'threshold': 0.75" in (tmp_path / "source.py").read_text()


def test_compile_logs_each_stage():
    lines = []
    handler = logging.Handler()
    handler.emit = lambda record: lines.append(record.getMessage())
    log = logging.getLogger("dspy.teleprompt.reanchor.reanchor")
    log.addHandler(handler)
    try:
        ReAnchor(metric, num_threads=2).compile(Decide(Sig), trainset=examples(), valset=examples("v-"))
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
        ReAnchor(metric).compile(Decide(Sig), trainset=[])


def test_a_student_without_a_decide_fails():
    with pytest.raises(ValueError, match="discoverable Decide"):
        ReAnchor(metric).compile(dspy.Predict("pair -> match: bool"), trainset=examples())
