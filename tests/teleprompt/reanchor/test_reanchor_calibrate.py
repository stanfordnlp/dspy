"""Calibration fits each predictor's thresholds, cuts, and weights against the metric, in place."""

from typing import Annotated, Literal

import pytest

import dspy
from dspy.experimental import Choice, Score
from dspy.teleprompt.reanchor.calibrate import calibrate, predictors
from tests.teleprompt.reanchor.fakes import choice, noul, score

Rating = Score["low", "mid", "high"]


class Match(dspy.Signature):
    pair: str = dspy.InputField()
    match: bool = dspy.OutputField(desc="Are the two the same?")


def test_a_leaning_noul_gets_a_threshold_between_its_piles(system_one):
    system_one(lambda state, name, q: noul(0.9 if state["inputs"]["pair"] == "same" else 0.7))
    program = dspy.Predict(Match)
    train = [dspy.Example(pair=k, match=k == "same").with_inputs("pair") for k in ["same", "different"] * 4]
    report = calibrate(program, train, lambda g, p, trace=None: float(p.match == g.match), num_threads=2)
    assert program.fields["match"]["threshold"] == 0.75
    assert report[0]["train_score_at_default"] == 0.5 and report[0]["train_score"] == 1.0


def test_a_starting_threshold_off_the_grid_stays_when_no_grid_value_beats_it(system_one):
    # 0.73 splits 0.74 from 0.72; no grid value does.
    system_one(lambda state, name, q: noul(0.74 if state["inputs"]["pair"] == "same" else 0.72))
    program = dspy.Predict(Match)
    program.fields["match"] = {"threshold": 0.73}
    train = [dspy.Example(pair=k, match=k == "same").with_inputs("pair") for k in ["same", "different"] * 4]
    report = calibrate(program, train, lambda g, p, trace=None: float(p.match == g.match), num_threads=2)
    assert program.fields["match"]["threshold"] == 0.73
    assert report[0]["train_score_at_start"] == 1.0 and report[0]["train_score"] == 1.0


class Kind(dspy.Signature):
    item: str = dspy.InputField()
    kind: Literal["x", "y"] = dspy.OutputField(desc="Which kind?")


def test_choice_multipliers_move_an_overpicked_option(system_one):
    # Jev leans to "x": 0.6 on an "x" item and 0.55 on a "y" item.
    system_one(
        lambda state, name, q: choice(
            {"x": 0.6, "y": 0.4} if state["inputs"]["item"] == "x" else {"x": 0.55, "y": 0.45}
        )
    )
    program = dspy.Predict(Kind)
    train = [dspy.Example(item=k, kind=k).with_inputs("item") for k in ["x", "y"] * 4]
    report = calibrate(program, train, lambda g, p, trace=None: float(p.kind == g.kind), num_threads=2)
    assert report[0]["train_score"] == 1.0
    w = program.fields["kind"]["weights"]
    assert 0.55 * w["x"] < 0.45 * w["y"] and 0.6 * w["x"] > 0.4 * w["y"]


# Jev leans high: a low item still puts half its mass on the top level (mean index 1.3).
LEANING = {"lo": {0: 0.2, 1: 0.3, 2: 0.5}, "hi": {0: 0.0, 1: 0.2, 2: 0.8}}


class RateLevel(dspy.Signature):
    item: str = dspy.InputField()
    rating: Rating = dspy.OutputField(desc="How good is it?")


def test_score_cuts_move_to_fit_a_metric_on_the_level(system_one):
    system_one(lambda state, name, q: score(LEANING[state["inputs"]["item"]]))
    program = dspy.Predict(RateLevel)
    train = [dspy.Example(item=k, rating={"lo": 0, "hi": 2}[k]).with_inputs("item") for k in ["lo", "hi"] * 3]
    report = calibrate(program, train, lambda g, p, trace=None: float(p.rating.level == g.rating), num_threads=2)
    cuts = program.fields["rating"]["cuts"]
    assert report[0]["parameter"] == "cuts"
    assert report[0]["train_score_at_start"] == 0.5 and report[0]["train_score"] == 1.0
    assert 1.3 < cuts[0] < cuts[1] <= 1.8


def test_score_cuts_stay_at_their_defaults_when_the_metric_ignores_the_level(system_one):
    system_one(lambda state, name, q: score(LEANING[state["inputs"]["item"]]))
    program = dspy.Predict(RateLevel)
    train = [dspy.Example(item=k, rating={"lo": 0, "hi": 2}[k]).with_inputs("item") for k in ["lo", "hi"]]
    report = calibrate(program, train, lambda g, p, trace=None: 1 - abs(p.rating.value - g.rating) / 2, num_threads=2)
    assert program.fields["rating"]["cuts"] == [0.5, 1.5]
    assert report == [{"predictor": "self", "field": "rating", "skipped": "the metric does not read this output"}]


class Route(dspy.Signature):
    ticket: str = dspy.InputField()
    queue: Annotated[Literal["billing", "tech"], Choice[("billing", "Payments"), ("tech", "Bugs")]] = dspy.OutputField(
        desc="Which queue?"
    )


def test_a_native_literal_with_choice_criteria_gets_weights_and_stays_native(system_one):
    system_one(lambda state, name, q: choice({"billing": 0.6, "tech": 0.4}))
    program = dspy.Predict(Route)
    train = [dspy.Example(ticket=str(i), queue="tech" if i % 3 else "billing").with_inputs("ticket") for i in range(6)]
    report = calibrate(program, train, lambda g, p, trace=None: float(p.queue == g.queue), num_threads=2)
    assert report[0]["parameter"] == "weights" and report[0]["train_score"] == round(4 / 6, 4)
    assert program(ticket="0").queue == "tech"


def test_predictors_finds_nested_and_listed_predictors():
    class Wrapper(dspy.Module):
        def __init__(self):
            super().__init__()
            self.judge = dspy.Predict(Match)
            self.judges = [dspy.Predict(Kind)]
            self.writer = dspy.Predict("pair -> note")

    assert [name for name, _ in predictors(Wrapper())] == ["judge", "judges[0]"]
    assert [name for name, _ in predictors(dspy.Predict(Match))] == ["self"]


@pytest.mark.parametrize("threads", [1, 4])
def test_calibration_leaves_the_request_unchanged(system_one, threads):
    client = system_one(lambda state, name, q: noul(0.9 if state["inputs"]["pair"] == "same" else 0.7))
    program = dspy.Predict(Match)
    train = [dspy.Example(pair=k, match=k == "same").with_inputs("pair") for k in ["same", "different"]]
    calibrate(program, train, lambda g, p, trace=None: float(p.match == g.match), num_threads=threads)
    assert (
        len({repr(c) for c in client.calls}) == 2
    )  # one distinct request per example, however many settings were tried


def test_a_metric_error_fails_the_pass(system_one):
    system_one(lambda state, name, q: noul(0.9))

    def broken(example, prediction):
        raise RuntimeError("metric failed")

    train = [dspy.Example(pair="same", match=True).with_inputs("pair")]
    with pytest.raises(Exception, match="Execution cancelled"):
        calibrate(dspy.Predict(Match), train, broken, num_threads=1)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_nonfinite_metric_fails(system_one, value):
    system_one(lambda state, name, q: noul(0.9))
    train = [dspy.Example(pair="same", match=True).with_inputs("pair")]
    with pytest.raises(ValueError, match="finite metric"):
        calibrate(dspy.Predict(Match), train, lambda g, p, trace=None: value, num_threads=1)


class TwoFlags(dspy.Signature):
    pair: str = dspy.InputField()
    match: bool = dspy.OutputField(desc="Are the two the same?")
    noise: bool = dspy.OutputField(desc="Is it noisy?")
    kind: Literal["x", "y"] = dspy.OutputField(desc="Which kind?")


def two_flags(state, name, q):
    if name == "kind":
        return choice({"x": 0.6, "y": 0.4})
    return noul(0.9 if state["inputs"]["pair"] == "same" else 0.7)


def test_outputs_the_metric_does_not_read_are_skipped(system_one):
    client = system_one(two_flags)
    program = dspy.Predict(TwoFlags)
    train = [dspy.Example(pair=k, match=k == "same").with_inputs("pair") for k in ["same", "different"] * 4]
    report = calibrate(program, train, lambda g, p, trace=None: float(p.match == g.match), num_threads=2)
    rows = {r["field"]: r for r in report}
    assert rows["noise"]["skipped"] == "the metric does not read this output"
    assert rows["kind"]["skipped"] == "the metric does not read this output"
    assert "skipped" not in rows["match"] and program.fields["match"]["threshold"] == 0.75
    assert "noise" not in program.fields and "kind" not in program.fields
    assert len({repr(c) for c in client.calls}) == 2  # probing the extremes asks nothing new


def test_balanced_labels_are_not_mistaken_for_an_ignored_output(system_one):
    # All True and all False both score 0.5 on average; the per-example scores still differ.
    system_one(lambda state, name, q: noul(0.9 if state["inputs"]["pair"] == "same" else 0.7))
    program = dspy.Predict(Match)
    train = [dspy.Example(pair=k, match=k == "same").with_inputs("pair") for k in ["same", "different"] * 2]
    report = calibrate(program, train, lambda g, p, trace=None: float(p.match == g.match), num_threads=2)
    assert "skipped" not in report[0] and program.fields["match"]["threshold"] == 0.75


def test_outputs_limits_the_fit(system_one):
    system_one(two_flags)
    program = dspy.Predict(TwoFlags)
    train = [dspy.Example(pair=k, match=k == "same").with_inputs("pair") for k in ["same", "different"] * 2]
    report = calibrate(
        program, train, lambda g, p, trace=None: float(p.match == g.match), num_threads=2, outputs={"noise"}
    )
    assert [r["field"] for r in report] == ["noise"] and "match" not in program.fields
