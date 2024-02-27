import dspy
from dspy.primitives.program import (
    Module,
    set_attribute_by_name,
)  # Adjust the import based on your file structure
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
    assert (
        module._compiled is False
    ), "Module _compiled attribute should be False upon initialization"


def test_named_predictors():
    module = HopModule()
    named_preds = module.named_predictors()
    assert len(named_preds) == 2, "Should identify correct number of Predict instances"
    names, preds = zip(*named_preds)
    assert (
        "predict1" in names and "predict2" in names
    ), "Named predictors should include 'predict1' and 'predict2'"


def test_predictors():
    module = HopModule()
    preds = module.predictors()
    assert len(preds) == 2, "Should return correct number of Predict instances"
    assert all(
        isinstance(p, dspy.Predict) for p in preds
    ), "All returned items should be instances of PredictMock"


def test_forward():
    program = HopModule()
    dspy.settings.configure(
        lm=DummyLM({"What is 1+1?": "let me check", "let me check": "2"})
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
    names, _preds = zip(*named_preds)
    assert "hop.predict1" in names
    assert "hop.predict2" in names
