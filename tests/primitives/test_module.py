import json
import tempfile
from pathlib import Path

import pytest

import dspy
from dspy.primitives.example import Example
from dspy.primitives.module import Module, _collect_predictions, set_attribute_by_name
from dspy.primitives.prediction import Prediction
from dspy.utils import DummyLM


class HopModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict1 = dspy.Predict("question -> query")
        self.predict2 = dspy.Predict("query -> answer")

    def forward(self, question):
        query = self.predict1(question=question).query
        return self.predict2(query=query)


def test_module_initialization():
    module = Module()
    assert module._compiled is False, "Module _compiled attribute should be False upon initialization"


def test_named_predictors():
    module = HopModule()
    named_preds = module.named_predictors()
    assert len(named_preds) == 2, "Should identify correct number of Predict instances"
    names, _preds = zip(*named_preds, strict=False)
    assert "predict1" in names and "predict2" in names, "Named predictors should include 'predict1' and 'predict2'"


def test_predictors():
    module = HopModule()
    preds = module.predictors()
    assert len(preds) == 2, "Should return correct number of Predict instances"
    assert all(isinstance(p, dspy.Predict) for p in preds), "All returned items should be instances of PredictMock"


def test_forward():
    program = HopModule()
    dspy.configure(
        lm=DummyLM(
            {
                "What is 1+1?": {"query": "let me check"},
                "let me check": {"answer": "2"},
            }
        )
    )
    result = program(question="What is 1+1?").answer
    assert result == "2"


def test_nested_named_predictors():
    class Hop2Module(dspy.Module):
        def __init__(self):
            super().__init__()
            self.hop = HopModule()

    module = Hop2Module()
    named_preds = module.named_predictors()
    assert len(named_preds) == 2
    names, _preds = zip(*named_preds, strict=False)
    assert "hop.predict1" in names
    assert "hop.predict2" in names


def test_empty_module():
    module = Module()
    assert list(module.named_sub_modules()) == [("self", module)]


def test_single_level():
    module = Module()
    module.sub = Module()
    expected = [("self", module), ("self.sub", module.sub)]
    assert list(module.named_sub_modules()) == expected


def test_multiple_levels():
    module = Module()
    module.sub = Module()
    module.sub.subsub = Module()
    expected = [("self", module), ("self.sub", module.sub), ("self.sub.subsub", module.sub.subsub)]
    assert list(module.named_sub_modules()) == expected


def test_multiple_sub_modules():
    module = Module()
    module.sub1 = Module()
    module.sub2 = Module()
    expected = [("self", module), ("self.sub1", module.sub1), ("self.sub2", module.sub2)]
    assert sorted(module.named_sub_modules()) == sorted(expected)


def test_non_base_module_attributes():
    module = Module()
    module.sub = Module()
    module.not_a_sub = "Not a self"
    expected = [("self", module), ("self.sub", module.sub)]
    assert list(module.named_sub_modules()) == expected


def test_complex_module_traversal():
    root = Module()
    root.sub_module = Module()
    root.sub_module.nested_list = [Module(), {"key": Module()}]
    root.sub_module.nested_tuple = (Module(), [Module(), Module()])
    expected_names = {
        "self",
        "self.sub_module",
        "self.sub_module.nested_list[0]",
        "self.sub_module.nested_list[1][key]",
        "self.sub_module.nested_tuple[0]",
        "self.sub_module.nested_tuple[1][0]",
        "self.sub_module.nested_tuple[1][1]",
    }
    found_names = {name for name, _ in root.named_sub_modules()}

    assert found_names == expected_names, (
        f"Missing or extra modules found. Missing: {expected_names - found_names}, Extra: {found_names - expected_names}"
    )


def test_complex_module_traversal_with_same_module():
    root = Module()
    root.sub_module = Module()
    root.sub_module.nested_list = [Module(), {"key": Module()}]
    same_module = Module()
    root.sub_module.nested_tuple = (Module(), [same_module, same_module])
    expected_names = {
        "self",
        "self.sub_module",
        "self.sub_module.nested_list[0]",
        "self.sub_module.nested_list[1][key]",  # NOTE: named_sub_modules allows recursive structures
        "self.sub_module.nested_tuple[0]",
        "self.sub_module.nested_tuple[1][0]",  # NEW: named_sub_modules allows recursive structures, but named_parameters does not
    }
    found_names = {name for name, _ in root.named_sub_modules()}

    assert found_names == expected_names, (
        f"Missing or extra modules found. Missing: {expected_names - found_names}, Extra: {found_names - expected_names}"
    )


def test_complex_module_set_attribute_by_name():
    root = Module()
    root.sub_module = Module()
    root.sub_module.nested_list = [Module(), {"key": Module()}]
    same_module = Module()
    root.sub_module.nested_tuple = (Module(), [same_module, same_module])

    set_attribute_by_name(root, "test_attrib", True)
    assert root.test_attrib is True
    set_attribute_by_name(root, "sub_module.test_attrib", True)
    assert root.sub_module.test_attrib is True
    set_attribute_by_name(root, "sub_module.nested_list[0].test_attrib", True)
    assert root.sub_module.nested_list[0].test_attrib is True
    set_attribute_by_name(root, "sub_module.nested_list[1]['key'].test_attrib", True)
    assert root.sub_module.nested_list[1]["key"].test_attrib is True
    set_attribute_by_name(root, "sub_module.nested_tuple[0].test_attrib", True)
    assert root.sub_module.nested_tuple[0].test_attrib is True
    set_attribute_by_name(root, "sub_module.nested_tuple[1][0].test_attrib", True)
    assert root.sub_module.nested_tuple[1][0].test_attrib is True
    assert root.sub_module.nested_tuple[1][1].test_attrib is True


class DuplicateModule(Module):
    def __init__(self):
        super().__init__()
        self.p0 = dspy.Predict("question -> answer")
        self.p1 = self.p0


def test_named_parameters_duplicate_references():
    module = DuplicateModule()
    # Only testing for whether exceptions are thrown or not
    # As Module.named_parameters() is recursive, this is mainly for catching infinite recursion
    module.named_parameters()


def test_load_dspy_program_cross_version():
    """
    Test backward compatibility for loading a saved DSPy program.

    This test verifies that DSPy can load a program saved in version 3.0.1, ensuring compatibility with older versions.
    The saved state is located in 'test/primitives/resources/saved_program.json' and represents an optimized
    `dspy.ReAct` program.
    """
    path = Path(__file__).parent / "resources" / "saved_program.json"
    loaded_react = dspy.ReAct("question->answer", tools=[])
    loaded_react.load(path)
    assert (
        "Imagine you are a detective racing against time to solve a high-profile"
        in loaded_react.react.signature.instructions
    )
    assert "Given the very verbose fields `question`" in loaded_react.extract.predict.signature.instructions

    assert len(loaded_react.react.demos) == 2
    assert len(loaded_react.extract.predict.demos) == 2

def test_load_state_is_transactional():
    """
    Regression test for https://github.com/stanfordnlp/dspy/issues/9589

    load_state must be all-or-nothing. If it fails mid-load (missing key
    or malformed value), the module must be completely unchanged.
    """

    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class Prog(dspy.Module):
        def __init__(self):
            super().__init__()
            self.a = dspy.ChainOfThought(Sig)
            self.b = dspy.ChainOfThought(Sig)

    source = Prog()
    sentinel = Example(question="q1", answer="a1").with_inputs("question")
    source.a.predict.demos = [sentinel]
    source.b.predict.demos = [sentinel]

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "state.json"
        source.save(str(path), save_program=False)

        raw = json.loads(path.read_text())
        corrupted = {k: v for k, v in raw.items() if "b." not in k}
        path.write_text(json.dumps(corrupted))

        template = Prog()
        assert template.a.predict.demos == []

        with pytest.raises(KeyError):
            template.load(str(path))

        assert template.a.predict.demos == [], (
            "load_state partially mutated module before failing"
        )


def test_collect_predictions_single():
    pred = Prediction(answer="hi")
    assert _collect_predictions(pred) == [pred]


def test_collect_predictions_tuple_with_trace():
    # Mirrors the (prediction, trace) shape used by GEPA bootstrap tracing.
    pred = Prediction(answer="hi")
    assert _collect_predictions((pred, {"trace": "data"})) == [pred]


def test_collect_predictions_list():
    p1 = Prediction(answer="a")
    p2 = Prediction(answer="b")
    assert _collect_predictions([p1, p2]) == [p1, p2]


def test_collect_predictions_nested_dict_and_list():
    p1 = Prediction(answer="a")
    p2 = Prediction(answer="b")
    p3 = Prediction(answer="c")
    output = {"main": p1, "others": [p2, [p3]]}
    collected = _collect_predictions(output)
    assert set(map(id, collected)) == {id(p1), id(p2), id(p3)}


def test_collect_predictions_no_predictions():
    assert _collect_predictions("just a string") == []
    assert _collect_predictions({"a": 1, "b": [2, 3]}) == []
    assert _collect_predictions(None) == []


def test_set_lm_usage_attaches_to_list_of_predictions():
    # Regression test for the dspy.Parallel-inside-Module case from #9201.
    # The forward returns a list of Predictions (as dspy.Parallel does) and
    # usage must be attached to every Prediction in the list.
    class ParallelModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.pred = dspy.Predict("q -> a")

        def forward(self, qs):
            return dspy.Parallel(num_threads=2).forward(
                [(self.pred, dspy.Example(q=q).with_inputs("q")) for q in qs]
            )

    lm = DummyLM([{"a": "1"}, {"a": "2"}])
    with dspy.context(lm=lm, track_usage=True):
        results = ParallelModule()(["x", "y"])

    assert len(results) == 2
    assert all(isinstance(r, Prediction) for r in results)
    assert all(r.get_lm_usage() is not None for r in results)


def test_set_lm_usage_attaches_to_nested_structure():
    class NestedModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.pred = dspy.Predict("q -> a")

        def forward(self):
            return {
                "main": self.pred(q="x"),
                "others": [self.pred(q="y"), self.pred(q="z")],
            }

    lm = DummyLM([{"a": "1"}, {"a": "2"}, {"a": "3"}])
    with dspy.context(lm=lm, track_usage=True):
        result = NestedModule()()

    assert isinstance(result, dict)
    predictions = [result["main"], *result["others"]]
    assert all(isinstance(p, Prediction) for p in predictions)
    assert all(p.get_lm_usage() is not None for p in predictions)


def test_set_lm_usage_preserves_single_prediction_behavior():
    # The most common case must continue to work unchanged.
    class SimpleModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.pred = dspy.Predict("q -> a")

        def forward(self, q):
            return self.pred(q=q)

    lm = DummyLM([{"a": "1"}])
    with dspy.context(lm=lm, track_usage=True):
        result = SimpleModule()(q="x")

    assert isinstance(result, Prediction)
    assert result.get_lm_usage() is not None


def test_set_lm_usage_no_warning_for_non_prediction_when_track_usage_disabled(monkeypatch):
    # When the user does not enable track_usage, _set_lm_usage is never
    # invoked and modules can return any type without warnings.
    warnings = []
    import dspy.primitives.module as module_mod

    monkeypatch.setattr(module_mod.logger, "warning", lambda msg, *a, **kw: warnings.append(msg))

    class StringModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.pred = dspy.Predict("q -> a")

        def forward(self, q):
            return self.pred(q=q).a

    with dspy.context(lm=DummyLM([{"a": "ok"}])):
        result = StringModule()(q="x")

    assert result == "ok"
    assert not any("Failed to set LM usage" in w for w in warnings)


def test_set_lm_usage_warns_when_no_prediction_found_with_track_usage(monkeypatch):
    # When track_usage is on but the module returns a container with no
    # Predictions, the warning must still fire so users notice the misuse.
    warnings = []
    import dspy.primitives.module as module_mod

    monkeypatch.setattr(module_mod.logger, "warning", lambda msg, *a, **kw: warnings.append(msg))

    class NoPredictionModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.pred = dspy.Predict("q -> a")

        def forward(self, q):
            return {"answer": self.pred(q=q).a}  # plain dict, no Prediction

    with dspy.context(lm=DummyLM([{"a": "ok"}]), track_usage=True):
        result = NoPredictionModule()(q="x")

    assert result == {"answer": "ok"}
    assert any("Failed to set LM usage" in w for w in warnings)
