"""The optimizer-facing surface of Predict, as used by custom predictors.

A custom predictor (e.g. one backed by an agent harness instead of the
LM/adapter path) subclasses ``dspy.Predict`` and overrides ``forward``.
Optimizers only rely on: discovery via ``named_predictors()``, the learned
parameters ``instructions``/``demos``, and ``(predictor, inputs, prediction)``
tuples in the trace. These tests pin that surface.
"""

import dspy
from dspy.utils.trace import record_trace


class BackendPredict(dspy.Predict):
    """A Predict whose execution bypasses the LM/adapter path entirely."""

    def forward(self, **kwargs):
        pred = dspy.Prediction(answer=f"backend saw: {self.instructions}")
        record_trace(self, kwargs, pred)
        return pred


class Program(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.Predict("question -> answer")
        self.backend = BackendPredict("question -> answer")


def test_named_predictors_discovers_predict_subclasses():
    program = Program()
    names = [name for name, _ in program.named_predictors()]
    assert names == ["qa", "backend"]


def test_instructions_is_the_canonical_parameter():
    pred = dspy.Predict("question -> answer")
    pred.instructions = "Answer concisely."
    assert pred.signature.instructions == "Answer concisely."
    assert pred.instructions == "Answer concisely."

    # Legacy signature reassignment stays observable through the property.
    pred.signature = pred.signature.with_instructions("Answer verbosely.")
    assert pred.instructions == "Answer verbosely."


def test_backend_predict_runs_without_lm_or_adapter():
    program = Program()
    program.backend.instructions = "Be brief."
    result = program.backend(question="hello")
    assert result.answer == "backend saw: Be brief."


def test_record_trace_matches_predict_tuple_shape():
    program = Program()
    with dspy.context(trace=[]):
        program.backend(question="hello")
        trace = dspy.settings.trace
        assert len(trace) == 1
        predictor, inputs, prediction = trace[0]
        assert predictor is program.backend
        assert inputs == {"question": "hello"}


def test_record_trace_bounds_trace_size():
    predictor = BackendPredict("question -> answer")
    with dspy.context(trace=[], max_trace_size=3):
        for i in range(5):
            record_trace(predictor, {"i": i}, dspy.Prediction(answer=str(i)))
        trace = dspy.settings.trace
        assert len(trace) == 3
        assert [t[1]["i"] for t in trace] == [2, 3, 4]


def test_record_trace_noop_when_disabled():
    predictor = BackendPredict("question -> answer")
    with dspy.context(trace=None):
        record_trace(predictor, {"q": "x"}, dspy.Prediction(answer="y"))
    with dspy.context(trace=[], max_trace_size=0):
        record_trace(predictor, {"q": "x"}, dspy.Prediction(answer="y"))
        assert dspy.settings.trace == []
