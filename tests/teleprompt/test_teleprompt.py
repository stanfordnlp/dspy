from dspy.teleprompt.teleprompt import Teleprompter


class DummyTeleprompter(Teleprompter):
    def __init__(self, param1: int, param2: str):
        super().__init__()
        self.param1 = param1
        self.param2 = param2

    def compile(self, student, *, trainset, teacher=None, valset=None, **kwargs):
        return student


def test_get_params():
    teleprompter = DummyTeleprompter(param1=1, param2="test")
    params = teleprompter.get_params()
    assert params == {"param1": 1, "param2": "test"}


def test_labeled_few_shot_empty_trainset_and_nosample():
    import dspy
    from dspy.teleprompt.vanilla import LabeledFewShot

    class DummyPredictor(dspy.Module):
        def __init__(self):
            super().__init__()
            self.pred = dspy.Predict("question -> answer")

    module = DummyPredictor()
    optimizer = LabeledFewShot(k=2)

    # Invariant 1: Empty trainset returns valid student module, not None
    compiled_empty = optimizer.compile(module, trainset=[])
    assert compiled_empty is not None, "compile with empty trainset must return student instance, not None"
    assert len(compiled_empty.pred.demos) == 0

    # Invariant 2: Deterministic slice when sample=False
    examples = [dspy.Example(question=str(i), answer=str(i)).with_inputs("question") for i in range(5)]
    compiled_slice = optimizer.compile(module, trainset=examples, sample=False)
    assert compiled_slice.pred.demos == examples[:2]

