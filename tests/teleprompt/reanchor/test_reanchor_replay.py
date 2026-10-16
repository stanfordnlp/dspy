"""ReAnchor replays raw answers to repeated requests within one calibration."""

import asyncio

import pytest

import dspy
from dspy.adapters.decision import replay_answers
from dspy.experimental import ReAnchor
from tests.teleprompt.reanchor.fakes import noul


def leaning(state, name, q):
    pair = state["inputs"]["pair"]
    if pair.endswith("same-tricky"):
        return noul(0.7)
    return noul(0.9 if "same" in pair else 0.7)


class Sig(dspy.Signature):
    pair: str = dspy.InputField()
    match: bool = dspy.OutputField(desc="Are the two the same?")


class Followup(dspy.Signature):
    pair: str = dspy.InputField()
    verdict: str = dspy.InputField()
    sure: bool = dspy.OutputField(desc="Is the verdict certain?")


@pytest.fixture(autouse=True)
def settings():
    dspy.configure(adapter=dspy.JSONAdapter())


def examples():
    kinds = ["same"] * 6 + ["same-tricky"] * 4 + ["different"] * 8
    return [dspy.Example(pair=k, match=k.startswith("same")).with_inputs("pair") for k in kinds]


def metric(gold, pred, trace=None):
    return float(pred.match == gold.match)


def compile_with(replay, program=None):
    optimizer = ReAnchor(metric, num_threads=2, replay=replay)
    compiled = optimizer.compile(program or dspy.Predict(Sig), trainset=examples())
    return compiled, optimizer.report


def test_replay_fits_the_same_settings_with_one_call_per_distinct_request(system_one):
    plain = system_one(leaning)
    expected_program, expected_report = compile_with(replay=False)
    replayed = system_one(leaning)
    program, report = compile_with(replay=True)
    assert program.fields == expected_program.fields and report == expected_report
    assert len(replayed.calls) == len({state["inputs"]["pair"] for state, _ in plain.calls}) == 3
    assert len(plain.calls) > 10 * len(replayed.calls)


def test_an_uncached_client_is_not_replayed(system_one):
    client = system_one(leaning, cache=False)
    ReAnchor(metric, num_threads=2, require_cache=False).compile(dspy.Predict(Sig), trainset=examples())
    with_replay = len(client.calls)
    client.calls.clear()
    ReAnchor(metric, num_threads=2, require_cache=False, replay=False).compile(dspy.Predict(Sig), trainset=examples())
    assert with_replay == len(client.calls)


class Chained(dspy.Module):
    """The second request's inputs depend on the first decision, so new thresholds produce new requests."""

    def __init__(self):
        super().__init__()
        self.judge = dspy.Predict(Sig)
        self.check = dspy.Predict(Followup)

    def forward(self, pair):
        match = self.judge(pair=pair).match
        sure = self.check(pair=pair, verdict="same" if match else "different").sure
        return dspy.Prediction(match=match and sure)


def chained(state, name, q):
    if name == "sure":
        return noul(0.8 if state["inputs"]["verdict"] == "same" else 0.6)
    return leaning(state, name, q)


def test_a_request_that_changes_with_an_earlier_decision_reaches_the_client(system_one):
    system_one(chained)
    expected_program, expected_report = compile_with(replay=False, program=Chained())
    client = system_one(chained)
    program, report = compile_with(replay=True, program=Chained())
    assert program.dump_state() == expected_program.dump_state() and report == expected_report
    verdicts = {(s["inputs"]["pair"], s["inputs"]["verdict"]) for s, _ in client.calls if "verdict" in s["inputs"]}
    assert ("same", "different") in verdicts and ("same", "same") in verdicts


def test_async_calls_are_replayed(system_one):
    client = system_one(leaning)
    predict = dspy.Predict(Sig)

    async def twice():
        return [await predict.acall(pair="same"), await predict.acall(pair="same")]

    with replay_answers():
        first, second = asyncio.run(twice())
    assert first.match == second.match and len(client.calls) == 1


class FlexibleSig(dspy.Signature):
    item: object = dspy.InputField()
    match: bool = dspy.OutputField(desc="Does this match?")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("first_input", "second_input"),
    [
        (True, 1),
        (1, 1.0),
        (0.0, -0.0),
        ({"a": 1}, [["a", 1]]),
        ({}, []),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),
    ],
)
def test_replay_distinguishes_request_values(system_one, first_input, second_input, asynchronous):
    client = system_one(lambda state, name, q: noul(0.9 if repr(state["inputs"]["item"]) == repr(first_input) else 0.1))
    predict = dspy.Predict(FlexibleSig)

    async def run():
        return [await predict.acall(item=item) for item in (first_input, second_input, first_input)]

    with replay_answers():
        results = (
            asyncio.run(run())
            if asynchronous
            else [predict(item=item) for item in (first_input, second_input, first_input)]
        )
    assert [result.match for result in results] == [True, False, True]
    assert len(client.calls) == 2


def generative_predict(*, cache_responses=True, cache=True):
    from dspy.utils.dummies import DummyLM

    client = DummyLM([{"match": noul(0.9)}, {"match": noul(0.1)}], adapter=dspy.JSONAdapter())
    client._cache_responses = cache_responses
    client.cache = cache
    dspy.configure(lm=client)
    predict = dspy.Predict(Sig)
    predict.fields = {"match": {}}
    return client, predict


@pytest.mark.parametrize("asynchronous", [False, True])
def test_response_cache_opt_out_is_respected(asynchronous):
    client, predict = generative_predict(cache_responses=False)

    async def run():
        return [await predict.acall(pair="same"), await predict.acall(pair="same")]

    with replay_answers():
        results = asyncio.run(run()) if asynchronous else [predict(pair="same"), predict(pair="same")]
    assert [result.match for result in results] == [True, False]
    assert len(client.history) == 2


@pytest.mark.parametrize("cache", [False, None, 0])
def test_call_level_cache_opt_out_is_respected(cache):
    client, predict = generative_predict()
    with replay_answers():
        results = [predict(pair="same", config={"cache": cache}) for _ in range(2)]
    assert [result.match for result in results] == [True, False]
    assert len(client.history) == 2


def test_call_level_cache_opt_in_allows_replay():
    client, predict = generative_predict(cache=False)
    with replay_answers():
        results = [predict(pair="same", config={"cache": True}) for _ in range(2)]
    assert all(result.match for result in results)
    assert len(client.history) == 1


@pytest.mark.parametrize("setting", ["temperature", "rollout_id", "model"])
def test_changed_lm_defaults_reach_the_client(setting):
    client, predict = generative_predict()
    client.kwargs["temperature"] = 0.7
    with replay_answers():
        first = predict(pair="same")
        if setting == "model":
            client.model = "another-model"
        else:
            client.kwargs[setting] = 1
        second = predict(pair="same")
        repeated = predict(pair="same")
    assert [first.match, second.match, repeated.match] == [True, False, False]
    assert len(client.history) == 2


def test_changed_adapter_reaches_the_client():
    client, predict = generative_predict()
    with replay_answers():
        first = predict(pair="same")
        adapter = dspy.ChatAdapter()
        client.adapter = adapter
        with dspy.context(adapter=adapter):
            second = predict(pair="same")
            repeated = predict(pair="same")
    assert [first.match, second.match, repeated.match] == [True, False, False]
    assert len(client.history) == 2


def test_changed_adapter_settings_reach_the_client():
    client, predict = generative_predict()
    adapter = dspy.settings.adapter
    with replay_answers():
        predict(pair="same")
        adapter.use_native_function_calling = False
        predict(pair="same")
        predict(pair="same")
    assert len(client.history) == 2


def test_opaque_mutable_inputs_skip_replay(system_one):
    class MutableInput:
        value = True

    item = MutableInput()
    client = system_one(lambda state, name, q: noul(0.9 if state["inputs"]["item"].value else 0.1))
    predict = dspy.Predict(FlexibleSig)
    with replay_answers():
        first = predict(item=item)
        item.value = False
        second = predict(item=item)
    assert [first.match, second.match] == [True, False]
    assert len(client.calls) == 2


def test_custom_adapters_use_the_normal_request_path():
    class StatefulAdapter:
        def __init__(self):
            self.probability = 0.9

        def __call__(self, lm, **request):
            return [{"match": noul(self.probability)}]

    _, predict = generative_predict()
    adapter = StatefulAdapter()
    with dspy.context(adapter=adapter), replay_answers():
        first = predict(pair="same")
        adapter.probability = 0.1
        second = predict(pair="same")
    assert [first.match, second.match] == [True, False]


def test_replay_keeps_runtime_clients_alive_until_the_context_exits():
    import gc
    import weakref

    from tests.teleprompt.reanchor.fakes import FakeClient

    predict = dspy.Predict(Sig)
    with replay_answers():
        client = FakeClient(leaning)
        reference = weakref.ref(client)
        with dspy.context(lm=client):
            predict(pair="same")
        del client
        gc.collect()
        assert reference() is not None  # An id in a replay key must not be reused by a new client.
    gc.collect()
    assert reference() is None


def test_generative_calibration_matches_without_replay():
    from tests.teleprompt.reanchor.fakes import ComputedLM

    def install():
        client = ComputedLM(
            lambda inputs, evidence: {"match": leaning({"inputs": inputs}, "match", {}) if evidence else False},
            adapter=dspy.JSONAdapter(),
        )
        client._cache_responses = True
        dspy.configure(lm=client)
        return client

    plain = install()
    expected, expected_report = compile_with(replay=False)
    replayed = install()
    program, report = compile_with(replay=True)
    assert program.fields == expected.fields
    assert report == expected_report
    assert len(replayed.history) < len(plain.history)


def test_replay_stores_are_independent(system_one):
    client = system_one(leaning)
    predict = dspy.Predict(Sig)
    with replay_answers():
        predict(pair="same")
        with replay_answers():
            predict(pair="same")
            predict(pair="same")
        predict(pair="same")
    predict(pair="same")
    assert len(client.calls) == 3


@pytest.mark.parametrize("asynchronous", [False, True])
def test_mutating_nondecision_outputs_does_not_change_replayed_answers(asynchronous):
    from dspy.utils.dummies import DummyLM

    class Mixed(dspy.Signature):
        pair: str = dspy.InputField()
        match: bool = dspy.OutputField(desc="Are the two the same?")
        reasons: list[str] = dspy.OutputField()

    client = DummyLM([{"match": noul(0.9), "reasons": ["original"]}], adapter=dspy.JSONAdapter())
    client._cache_responses = True
    dspy.configure(lm=client)
    predict = dspy.Predict(Mixed)
    predict.fields = {"match": {}}

    async def run():
        results = []
        for _ in range(3):
            result = await predict.acall(pair="same")
            results.append(list(result.reasons))
            result.reasons.append("mutated")
        return results

    with replay_answers():
        if asynchronous:
            results = asyncio.run(run())
        else:
            results = []
            for _ in range(3):
                result = predict(pair="same")
                results.append(list(result.reasons))
                result.reasons.append("mutated")
    assert results == [["original"]] * 3
    assert len(client.history) == 1
