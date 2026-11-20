import dspy
from dspy import Predict
from dspy.teleprompt.simba_utils import append_a_demo


def test_append_a_demo_with_overlapping_input_output_keys():
    # Same root cause as #9472 in SIMBA's demo appending: when a trace step's
    # recorded inputs carry a key that is also a predicted output (e.g. an
    # output field listed in `with_inputs()`), building the demo with
    # `dspy.Example(augmented=True, **inputs, **outputs)` raised
    # "got multiple values for keyword argument".
    class SimpleModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = Predict("context, turn -> category")

        def forward(self, **kwargs):
            return self.predictor(**kwargs)

    program = SimpleModule()
    predictor = program.predictor

    trace = [
        (predictor, {"context": "c", "turn": "t", "category": "echoed"}, {"category": "predicted"}),
    ]
    append_a_demo(demo_input_field_maxlen=None)(
        [{"score": 1.0, "trace": trace}],
        system=program,
        predictor2name={id(predictor): "predictor"},
        name2predictor={"predictor": predictor},
        batch_10p_score=0.0,
    )

    assert len(predictor.demos) == 1
    # The freshly predicted value wins for a key that appears in both inputs and outputs.
    assert predictor.demos[0].category == "predicted"
