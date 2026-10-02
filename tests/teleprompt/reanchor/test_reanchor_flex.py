"""ReAnchor on dspy.Flex programs: calibrating the predictors a Flex's code builds, and rewriting that code.

A Flex builds its predictors on every forward, so ReAnchor finds them by running the program and
stores what it fits in the Flex's ``predictor_fields``. With a proposer, it also replaces the Flex's
code with calibrated rewrites that score higher. The proposer itself (an RLM) is stubbed here; the
rewrites are fixed sources. Flex code runs in ``dspy.LocalInterpreter``.
"""

import json
import logging
import textwrap

import pytest

import dspy
from dspy.experimental import ReAnchor
from dspy.teleprompt.reanchor import decompose
from dspy.teleprompt.reanchor.calibrate import calibrate
from dspy.utils.exceptions import AdapterParseError
from tests.teleprompt.reanchor.fakes import ComputedLM, noul


class Sig(dspy.Signature):
    pair: str = dspy.InputField()
    match: bool = dspy.OutputField(desc="Are the two the same?")


def leaning(state, name, q):
    """The same evidence as test_reanchor.py: the best threshold is 0.8, taking train from 0.5556 to 0.7778."""
    pair = state["inputs"]["pair"]
    if q["instructions"] == "Is this a known lookalike?":
        return noul(0.95 if pair.endswith("tricky") else 0.05)
    if pair.endswith("same-tricky"):
        return noul(0.7)
    return noul(0.9 if "same" in pair else 0.7)


@pytest.fixture(autouse=True)
def jev(system_one):
    dspy.configure(adapter=dspy.JSONAdapter())
    return system_one(leaning)


def examples(prefix=""):
    kinds = ["same"] * 6 + ["same-tricky"] * 4 + ["different"] * 8
    return [dspy.Example(pair=prefix + k, match=k.startswith("same")).with_inputs("pair") for k in kinds]


def metric(gold, pred, trace=None):
    return float(pred.match == gold.match)


def flex(source=None, signature=Sig):
    program = dspy.Flex(signature, interpreter_factory=dspy.LocalInterpreter)
    if source is not None:
        program._bind_code(textwrap.dedent(source).strip())
    return program


def test_a_flex_baseline_calibrates_like_the_predict_it_wraps(jev):
    student = flex()
    optimizer = ReAnchor(metric, num_threads=4)
    program = optimizer.compile(student, trainset=examples())
    assert program.predictor_fields == {"predict": {"match": {"threshold": 0.8}}}
    assert optimizer.report["train_score_before"] == 0.5556 and optimizer.report["train_score"] == 0.7778
    assert optimizer.report["fitted"][0]["predictor"] == "predict"
    assert student.predictor_fields == {} and program._compiled
    flex_requests = {json.dumps(call, sort_keys=True) for call in jev.calls}

    jev.calls.clear()
    predict = ReAnchor(metric, num_threads=4).compile(dspy.Predict(Sig), trainset=examples())
    assert predict.fields == program.predictor_fields["predict"]
    assert {json.dumps(call, sort_keys=True) for call in jev.calls} == flex_requests  # Identical requests.


GATED = """
class Gated(dspy.Module):
    def __init__(self):
        super().__init__()
        self.judge = dspy.Predict("pair -> same: bool, lookalike: bool")
        self.judge.fields["same"] = {"instructions": "Are the two the same?"}
        self.judge.fields["lookalike"] = {"instructions": "Is this a known lookalike?"}

    def forward(self, pair):
        out = self.judge(pair=pair)
        return dspy.Prediction(match=bool(out.same) or bool(out.lookalike))
"""


def test_every_decision_of_a_decomposed_flex_is_calibrated():
    optimizer = ReAnchor(metric, num_threads=4)
    program = optimizer.compile(flex(GATED), trainset=examples())
    # "lookalike" catches the tricky pairs, so "same" can move up to reject the different ones.
    assert optimizer.report["train_score"] == 1.0
    rows = {row["field"]: row for row in optimizer.report["fitted"]}
    assert rows["same"]["predictor"] == "judge" and rows["same"]["value"] == 0.8
    assert program.predictor_fields["judge"]["same"]["threshold"] == 0.8
    # The code's question is kept alongside the fitted threshold.
    assert program.predictor_fields["judge"]["same"]["instructions"] == "Are the two the same?"


def test_a_flex_inside_a_module_is_named_by_its_path():
    class Outer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.checker = flex()

        def forward(self, pair):
            return self.checker(pair=pair)

    optimizer = ReAnchor(metric, num_threads=4)
    program = optimizer.compile(Outer(), trainset=examples())
    assert optimizer.report["fitted"][0]["predictor"] == "checker.predict"
    assert program.checker.predictor_fields == {"predict": {"match": {"threshold": 0.8}}}


def test_an_output_without_a_question_is_skipped_rather_than_failing_the_run():
    source = """
    class Bare(dspy.Module):
        def __init__(self):
            super().__init__()
            self.judge = dspy.Predict("pair -> same: bool")

        def forward(self, pair):
            return dspy.Prediction(match=self.judge(pair=pair).same)
    """
    # The generative path answers a bare bool natively; enabling its evidence would need a question.
    dspy.configure(adapter=dspy.ChatAdapter())
    lm = ComputedLM(lambda inputs, evidence: {"same": True})
    student = flex(source)
    student.lm = lm
    optimizer = ReAnchor(metric, num_threads=1, require_cache=False)
    program = optimizer.compile(student, trainset=examples())
    [row] = optimizer.report["fitted"]
    assert row["predictor"] == "judge" and "requires an OutputField" in row["skipped"]
    assert program.predictor_fields == {}


def test_a_flex_that_calls_no_decision_predictor_is_rejected():
    source = """
    class Rule(dspy.Module):
        def __init__(self):
            super().__init__()

        def forward(self, pair):
            return dspy.Prediction(match=pair.startswith("same"))
    """
    with pytest.raises(ValueError, match="No predictor with a decision output ran"):
        ReAnchor(metric, num_threads=2).compile(flex(source), trainset=examples())


def test_an_uncached_client_is_refused_before_the_flex_runs(system_one):
    system_one(leaning, cache=False)
    with pytest.raises(ValueError, match=r"Predictors \['self'\] do not cache"):
        ReAnchor(metric).compile(flex(), trainset=examples())


def test_a_predictor_built_with_different_outputs_is_not_calibrated(caplog):
    source = """
    class Shifting(dspy.Module):
        def __init__(self):
            super().__init__()

        def forward(self, pair):
            output = "same" if "same" in pair else "differs"
            self.judge = dspy.Predict(f"pair -> {output}: bool")
            self.judge.fields[output] = {"instructions": "Are the two the same?"}
            return dspy.Prediction(match=bool(getattr(self.judge(pair=pair), output)))
    """
    with caplog.at_level(logging.WARNING):
        report = calibrate(flex(source), examples(), metric, num_threads=2)
    assert report == []
    assert "'judge' was built with different decision outputs" in caplog.text


def test_calibrating_leaves_no_empty_predictor_fields_entries():
    program = flex(GATED)
    program.predictor_fields = {"judge": {}}
    calibrate(program, examples()[:6], metric, num_threads=2)  # "same" pairs only: nothing beats the default.
    assert program.predictor_fields == {}


# Decomposition. `propose` is replaced by a queue of fixed rewrites; each call records what it was given.


@pytest.fixture
def proposals(monkeypatch):
    queue, calls = [], []

    def fake_propose(flex, proposer, attempts, records, trainset, metric, max_iters=20, num_threads=None, **where):
        calls.append({"flex": flex, "attempts": attempts, "records": records})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr("dspy.teleprompt.reanchor.reanchor.propose", fake_propose)
    return queue, calls


BROKEN = """
class Broken(dspy.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pair):
        return dspy.Prediction(match=undefined_name)
"""


def test_a_better_decomposition_replaces_the_code_and_is_calibrated(proposals):
    queue, calls = proposals
    queue.extend([BROKEN, GATED])
    optimizer = ReAnchor(metric, num_threads=4, proposer=dspy.utils.DummyLM([]), rounds=2)
    val = examples("v-")
    program = optimizer.compile(flex(), trainset=examples(), valset=val)

    assert "class Gated" in program.module_src
    assert program.predictor_fields["judge"]["same"]["threshold"] == 0.8
    assert optimizer.report["val_score"] == optimizer.report["train_score"] == 1.0
    baseline, broken, gated = optimizer.report["decomposition"]
    assert baseline["accepted"] and baseline["score"] == 0.7778 and baseline["calls"] == 1.0
    assert not broken["accepted"] and "NameError" in broken["error"] and "undefined_name" in broken["error"]
    assert gated["accepted"] and gated["score"] == 1.0 and gated["round"] == 2

    # The proposer reads the calibrated program's results, and the failure it caused last round.
    first, second = calls
    assert first["records"][6]["outputs"] == {"match": False} and first["records"][6]["score"] == 0.0  # same-tricky
    assert first["records"][0]["decisions"] == [{"predictor": "predict", "output": "match", "evidence": {"noul": 0.9}}]
    assert ["NameError" in a.get("error", "") for a in second["attempts"]] == [False, True]


def test_a_proposer_that_fails_fails_the_round_not_the_run(proposals):
    queue, calls = proposals
    queue.extend([AdapterParseError("JSONAdapter", dspy.Signature("task -> module_src"), lm_response=""), GATED])
    optimizer = ReAnchor(metric, num_threads=4, proposer=dspy.utils.DummyLM([]), rounds=2)
    program = optimizer.compile(flex(), trainset=examples(), valset=examples("v-"))

    assert "class Gated" in program.module_src
    _, failed, gated = optimizer.report["decomposition"]
    assert not failed["accepted"] and "AdapterParseError" in failed["error"] and "source" not in failed
    assert gated["accepted"] and gated["round"] == 2
    assert ["AdapterParseError" in a.get("error", "") for a in calls[1]["attempts"]] == [False, True]


def test_a_decomposition_that_scores_lower_on_the_valset_is_not_kept(proposals):
    queue, _ = proposals
    always = """
    class Always(dspy.Module):
        def __init__(self):
            super().__init__()
            self.judge = dspy.Predict("pair -> same: bool")
            self.judge.fields["same"] = {"instructions": "Are the two the same?"}

        def forward(self, pair):
            self.judge(pair=pair)
            return dspy.Prediction(match=True)
    """
    queue.append(always)
    optimizer = ReAnchor(metric, num_threads=4, proposer=dspy.utils.DummyLM([]), rounds=1)
    student = flex()
    program = optimizer.compile(student, trainset=examples(), valset=examples("v-"))
    assert program.module_src == student.module_src
    assert program.predictor_fields == {"predict": {"match": {"threshold": 0.8}}}
    assert [a["accepted"] for a in optimizer.report["decomposition"]] == [True, False]


def test_an_equal_score_with_fewer_calls_is_kept(proposals):
    two_calls = """
    class Twice(dspy.Module):
        def __init__(self):
            super().__init__()
            self.first = dspy.Predict("pair -> same: bool")
            self.first.fields["same"] = {"instructions": "Are the two the same?"}
            self.second = dspy.Predict("pair -> same: bool")
            self.second.fields["same"] = {"instructions": "Are the two the same?"}

        def forward(self, pair):
            self.first(pair=pair)
            return dspy.Prediction(match=self.second(pair=pair).same)
    """
    queue, calls = proposals
    queue.extend([GATED, BROKEN])
    student = flex(two_calls)
    optimizer = ReAnchor(metric, num_threads=4, proposer=dspy.utils.DummyLM([]), rounds=2)
    program = optimizer.compile(student, trainset=examples()[:6])  # all "same": both codes score 1.0
    baseline, gated, broken = optimizer.report["decomposition"]
    assert baseline["calls"] == 2.0
    assert "class Gated" in program.module_src and gated["calls"] == 1.0
    # The proposer reads the best attempt first: the equal score with fewer calls, then the baseline.
    assert ["class Gated" in a["source"]["self"] for a in calls[1]["attempts"]] == [True, False]
    # Every attempt records its code the same way, by the Flex's path.
    assert all(set(a["source"]) == {"self"} for a in (baseline, gated, broken))


def test_without_a_valset_the_choice_is_made_on_the_trainset_with_a_warning(proposals, caplog):
    queue, _ = proposals
    queue.append(GATED)
    with caplog.at_level(logging.WARNING):
        program = ReAnchor(metric, num_threads=4, proposer=dspy.utils.DummyLM([]), rounds=1).compile(
            flex(), trainset=examples()
        )
    assert "class Gated" in program.module_src
    assert "pass a valset" in caplog.text


def test_the_proposer_is_not_used_for_a_program_without_a_flex(proposals):
    program = ReAnchor(metric, num_threads=2, proposer=dspy.utils.DummyLM([])).compile(
        dspy.Predict(Sig), trainset=examples()
    )
    assert program.fields == {"match": {"threshold": 0.8}}
    assert proposals[1] == []


def test_propose_runs_drafts_inside_the_whole_program(monkeypatch):
    class Outer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.checker = flex()

            self.gate = dspy.Predict("pair -> same: bool")  # Unbound: runs on the configured client.
            self.gate.fields["same"] = {"instructions": "Are the two the same?"}

        def forward(self, text):  # The program's input is not the Flex's.
            self.gate(pair=text)
            return self.checker(pair=text)

    captured = {}

    class FakeRLM:
        def __init__(self, signature, max_iters, tools, sub_lm):
            captured["tools"] = tools

        def __call__(self, **inputs):
            [run_code] = captured["tools"]
            captured["draft"] = json.loads(run_code(module_src=GATED, indexes=[0]))
            return dspy.Prediction(module_src=GATED)

    monkeypatch.setattr(dspy, "RLM", FakeRLM)
    program = Outer()
    data = [dspy.Example(text=e.pair, match=e.match).with_inputs("text") for e in examples()]
    decompose.propose(program.checker, dspy.utils.DummyLM([]), [], [], data, metric, program=program, path="checker")

    [draft] = captured["draft"]  # The proposer (a DummyLM with no answers) never serves the gate.
    assert "error" not in draft and draft["score"] == 1.0
    assert {d["predictor"] for d in draft["decisions"]} == {"gate", "judge"}
    assert "class Gated" not in program.checker.module_src  # Drafts run on a copy.


def test_propose_drives_an_rlm_whose_drafts_run_on_the_flex_backend(monkeypatch, jev):
    captured = {}

    class FakeRLM:
        def __init__(self, signature, max_iters, tools, sub_lm):
            captured.update(signature=signature, max_iters=max_iters, tools=tools, sub_lm=sub_lm)

        def __call__(self, **inputs):
            captured["inputs"] = inputs
            captured["lm"] = dspy.settings.lm
            [run_code] = captured["tools"]
            captured["draft"] = json.loads(run_code(module_src=GATED, indexes=[0, 6, 99]))
            captured["broken"] = json.loads(run_code(module_src="class Nope(:", indexes=[0]))
            return dspy.Prediction(module_src=f"```python\n{GATED.strip()}\n```")

    monkeypatch.setattr(dspy, "RLM", FakeRLM)
    proposer = dspy.utils.DummyLM([])
    source = decompose.propose(flex(), proposer, [{"source": "..."}], [{"index": 0}], examples(), metric)

    assert source == GATED.strip()
    assert captured["lm"] is proposer and captured["sub_lm"] is proposer
    assert captured["signature"] is decompose.DecomposeSignature
    assert captured["inputs"]["backend"] == decompose.SYSTEM_ONE_NOTE  # The Flex runs on a System One client.
    assert "Decision outputs" in captured["inputs"]["catalog"]
    # Drafts call the Flex's client, not the proposer, and report the evidence of every question.
    assert [r["index"] for r in captured["draft"]] == [0, 6]
    assert captured["draft"][1]["decisions"][1] == {
        "predictor": "judge",
        "output": "lookalike",
        "evidence": {"noul": 0.95},
    }
    assert "SyntaxError" in captured["broken"]["error"]
