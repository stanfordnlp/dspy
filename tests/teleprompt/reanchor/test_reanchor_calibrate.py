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
    assert program.fields["match"]["threshold"] == 0.8
    assert report[0]["train_score"] == 1.0


@pytest.mark.parametrize("target", [True, False])
def test_zero_probability_can_be_classified_on_either_side(system_one, target):
    system_one(lambda state, name, q: noul(0.0))
    program = dspy.Predict(Match)
    train = [dspy.Example(pair=str(i), match=target).with_inputs("pair") for i in range(10)]
    report = calibrate(program, train, lambda g, p: float(p.match == g.match), num_threads=2)
    assert program.fields["match"]["threshold"] == (0.0 if target else 0.5)
    assert report[0]["train_score"] == 1.0


def test_a_starting_threshold_stays_unless_a_candidate_scores_strictly_better(system_one):
    # 0.73 already splits 0.74 from 0.72, so the candidate in that gap only ties.
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
    assert report[0]["train_score"] == report[0]["train_score_at_start"]


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


def test_outputs_the_metric_does_not_read_keep_their_numeric_settings(system_one):
    client = system_one(two_flags)
    program = dspy.Predict(TwoFlags)
    train = [dspy.Example(pair=k, match=k == "same").with_inputs("pair") for k in ["same", "different"] * 4]
    report = calibrate(program, train, lambda g, p, trace=None: float(p.match == g.match), num_threads=2)
    rows = {r["field"]: r for r in report}
    assert rows["noise"]["train_score"] == rows["noise"]["train_score_at_start"]
    assert rows["kind"]["train_score"] == rows["kind"]["train_score_at_start"]
    assert "skipped" not in rows["match"] and program.fields["match"]["threshold"] == 0.8
    assert program.fields["noise"]["threshold"] == 0.5
    assert program.fields["kind"]["weights"] == {"x": 1.0, "y": 1.0}
    assert len({repr(c) for c in client.calls}) == 2


def test_balanced_labels_are_not_mistaken_for_an_ignored_output(system_one):
    # All True and all False both score 0.5 on average; the per-example scores still differ.
    system_one(lambda state, name, q: noul(0.9 if state["inputs"]["pair"] == "same" else 0.7))
    program = dspy.Predict(Match)
    train = [dspy.Example(pair=k, match=k == "same").with_inputs("pair") for k in ["same", "different"] * 2]
    report = calibrate(program, train, lambda g, p, trace=None: float(p.match == g.match), num_threads=2)
    assert "skipped" not in report[0] and program.fields["match"]["threshold"] == 0.8


def test_equal_scores_at_extremes_do_not_hide_a_useful_threshold(system_one):
    system_one(lambda state, name, q: noul(0.4 if state["inputs"]["pair"] == "same" else 0.2))

    class Batch(dspy.Module):
        def __init__(self):
            super().__init__()
            self.judge = dspy.Predict(Match)

        def forward(self, pairs):
            return dspy.Prediction(matches=[self.judge(pair=p).match for p in pairs])

    program = Batch()
    train = [dspy.Example(pairs=["same", "different"], matches=[True, False]).with_inputs("pairs") for _ in range(10)]
    # Default, all-True, and all-False decisions all fail exact match. Only the interior split wins.
    report = calibrate(program, train, lambda g, p, trace=None: float(p.matches == g.matches), num_threads=2)
    assert program.judge.fields["match"]["threshold"] == pytest.approx(0.3)
    assert report[0]["train_score_at_start"] == 0.0
    assert report[0]["train_score"] == 1.0


def test_probabilities_bunched_near_one_get_a_threshold_between_them(system_one):
    # Every P(True) is 0.98 or 1.0, above any fixed grid; the gap between them splits the labels.
    system_one(lambda state, name, q: noul(1.0 if state["inputs"]["pair"] == "same" else 0.98))
    program = dspy.Predict(Match)
    train = [dspy.Example(pair=k, match=k == "same").with_inputs("pair") for k in ["same", "different"] * 4]
    report = calibrate(program, train, lambda g, p, trace=None: float(p.match == g.match), num_threads=2)
    assert program.fields["match"]["threshold"] == 0.99
    assert report[0]["train_score"] == 1.0
    assert report[0]["observed"] == {"calls": 8, "distinct": 2, "min": 0.98, "max": 1.0, "candidates": 2}


def test_a_choice_multiplier_can_leave_the_range_a_fixed_grid_would_try(system_one):
    # A "y" item still puts 0.95 on "x"; separating the items needs a "y" multiplier above 19.
    system_one(
        lambda state, name, q: choice(
            {"x": 0.99, "y": 0.01} if state["inputs"]["item"] == "x" else {"x": 0.95, "y": 0.05}
        )
    )
    program = dspy.Predict(Kind)
    train = [dspy.Example(item=k, kind=k).with_inputs("item") for k in ["x", "y"] * 4]
    report = calibrate(program, train, lambda g, p, trace=None: float(p.kind == g.kind), num_threads=2)
    w = program.fields["kind"]["weights"]
    assert report[0]["train_score"] == 1.0
    assert 0.95 * w["x"] < 0.05 * w["y"] and 0.99 * w["x"] > 0.01 * w["y"]


def test_many_distinct_probabilities_are_thinned_to_the_candidate_cap(system_one):
    from dspy.teleprompt.reanchor.calibrate import MAX_CANDIDATES

    system_one(lambda state, name, q: noul(int(state["inputs"]["pair"]) / 200))
    program = dspy.Predict(Match)
    train = [dspy.Example(pair=str(i), match=i >= 120).with_inputs("pair") for i in range(200)]
    report = calibrate(program, train, lambda g, p, trace=None: float(p.match == g.match), num_threads=4)
    assert report[0]["observed"]["distinct"] == 200
    assert report[0]["observed"]["candidates"] <= MAX_CANDIDATES + 1  # Gap midpoints plus the zero endpoint.
    assert 0.55 < program.fields["match"]["threshold"] <= 0.6 and report[0]["train_score"] >= 0.97


def test_evidence_is_recorded_for_the_predictor_that_decoded_it(system_one):
    from dspy.adapters.decision import record_evidence

    system_one(lambda state, name, q: noul(0.9) if name == "match" else choice({"x": 0.6, "y": 0.4}))

    class Two(dspy.Module):
        def __init__(self):
            super().__init__()
            self.judge = dspy.Predict(Match)
            self.kind = dspy.Predict(Kind)

        def forward(self, items):
            return [self.judge(pair=i).match for i in items], self.kind(item="a").kind

    program = Two()
    with record_evidence() as log:
        program(items=["a", "b", "c"])
    program(items=["d"])
    assert [(caller is program.judge, name) for caller, name, _ in log].count((True, "match")) == 3
    caller, name, evidence = log[-1]
    assert caller is program.kind and name == "kind" and evidence["probabilities"] == {"x": 0.6, "y": 0.4}
    assert len(log) == 4


def test_concurrent_evidence_collectors_are_isolated_and_reach_dspy_workers(system_one):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from dspy.adapters.decision import record_evidence
    from dspy.utils.parallelizer import ParallelExecutor

    system_one(lambda state, name, q: noul(0.2 if state["inputs"]["pair"] == "a" else 0.8))
    barrier = Barrier(2)

    def collect(pair):
        predict = dspy.Predict(Match)
        with record_evidence() as log:
            barrier.wait(timeout=10)
            ParallelExecutor(num_threads=2, disable_progress_bar=True).execute(lambda _: predict(pair=pair), range(2))
            barrier.wait(timeout=10)
        return predict, log

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(collect, ["a", "b"]))
    for (predict, log), probability in zip(results, [0.2, 0.8], strict=True):
        assert len(log) == 2
        assert all(
            caller is predict and name == "match" and evidence == {"noul": probability}
            for caller, name, evidence in log
        )


def test_nested_evidence_collector_restores_outer_context_after_error(system_one):
    from dspy.adapters.decision import record_evidence

    system_one(lambda state, name, q: noul(float(state["inputs"]["pair"])))
    predict = dspy.Predict(Match)
    with record_evidence() as outer:
        predict(pair="0.1")
        with pytest.raises(RuntimeError, match="stop"):
            with record_evidence() as inner:
                predict(pair="0.2")
                raise RuntimeError("stop")
        predict(pair="0.3")
    predict(pair="0.4")
    assert [evidence["noul"] for _, _, evidence in outer] == [0.1, 0.3]
    assert [evidence["noul"] for _, _, evidence in inner] == [0.2]
    assert dspy.settings.get("_decision_evidence") is None


def lone_outlier(same: float, different: float, odd: float, count: int = 1):
    """20 same pairs, 19 different pairs, and `count` of the same pairs replaced by `odd` ones."""
    pairs = (
        [f"same{i}" for i in range(20 - count)] + [f"odd{i}" for i in range(count)] + [f"diff{i}" for i in range(19)]
    )

    def answer(state, name, q):
        pair = state["inputs"]["pair"]
        return noul(odd if pair.startswith("odd") else same if pair.startswith("same") else different)

    train = [dspy.Example(pair=p, match=not p.startswith("diff")).with_inputs("pair") for p in pairs]
    return answer, train


def test_a_threshold_stays_when_its_whole_gain_is_one_example(system_one):
    answer, train = lone_outlier(0.9, 0.1, 0.3)
    system_one(answer)
    program = dspy.Predict(Match)
    report = calibrate(program, train, lambda g, p, trace=None: float(p.match == g.match), num_threads=2)
    assert program.fields["match"]["threshold"] == 0.5
    assert report[0]["fold_check"] == {"passed": 0, "failed": 1}


def test_a_threshold_moves_when_its_gain_recurs_across_the_training_set(system_one):
    answer, train = lone_outlier(0.9, 0.1, 0.3, count=6)
    system_one(answer)
    program = dspy.Predict(Match)
    report = calibrate(program, train, lambda g, p, trace=None: float(p.match == g.match), num_threads=2)
    assert program.fields["match"]["threshold"] == 0.2
    assert report[0]["fold_check"] == {"passed": 1, "failed": 0}


def test_score_cuts_stay_when_their_whole_gain_is_one_example(system_one):
    levels = {"lo": {0: 0.8, 1: 0.1, 2: 0.1}, "hi": {0: 0.1, 1: 0.0, 2: 0.9}, "odd": {0: 0.5, 1: 0.3, 2: 0.2}}
    system_one(lambda state, name, q: score(levels[state["inputs"]["item"].rstrip("0123456789")]))
    program = dspy.Predict(RateLevel)
    items = [f"lo{i}" for i in range(19)] + ["odd0"] + [f"hi{i}" for i in range(20)]
    train = [dspy.Example(item=k, rating=2 if k.startswith("hi") else 0).with_inputs("item") for k in items]
    report = calibrate(program, train, lambda g, p, trace=None: float(p.rating.level == g.rating), num_threads=2)
    assert program.fields["rating"]["cuts"] == [0.5, 1.5]
    assert report[0]["fold_check"]["failed"] >= 1


def test_choice_weights_stay_when_their_whole_gain_is_one_example(system_one):
    leans = {"x": {"x": 0.8, "y": 0.2}, "y": {"x": 0.2, "y": 0.8}, "odd": {"x": 0.55, "y": 0.45}}
    system_one(lambda state, name, q: choice(leans[state["inputs"]["item"].rstrip("0123456789")]))
    program = dspy.Predict(Kind)
    items = [f"x{i}" for i in range(20)] + [f"y{i}" for i in range(19)] + ["odd0"]
    train = [dspy.Example(item=k, kind="x" if k.startswith("x") else "y").with_inputs("item") for k in items]
    report = calibrate(program, train, lambda g, p, trace=None: float(p.kind == g.kind), num_threads=2)
    assert program.fields["kind"]["weights"] == {"x": 1.0, "y": 1.0}
    assert report[0]["fold_check"]["failed"] >= 1
