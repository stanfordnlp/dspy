import dspy
import threading
from dspy.utils.dummies import DummyLM


def test_deepcopy_basic():
    signature = dspy.Signature("q -> a")
    cot = dspy.ChainOfThought(signature)
    cot_copy = cot.deepcopy()
    assert len(cot.parameters()) == len(cot_copy.parameters())
    # Parameters should be different objects with the same values.
    assert id(cot.parameters()[0]) != id(cot_copy.parameters()[0])
    assert cot.parameters()[0].__dict__ == cot_copy.parameters()[0].__dict__


def test_deepcopy_with_uncopyable_modules():
    class CustomClass(dspy.Module):
        def __init__(self):
            self.lock = threading.Lock()  # Non-copyable object.
            self.cot = dspy.ChainOfThought(dspy.Signature("q -> a"))

    model = CustomClass()
    model_copy = model.deepcopy()
    assert len(model.parameters()) == len(model_copy.parameters())
    # The lock should be refer to the same object (shallow copy).
    assert id(model.lock) == id(model_copy.lock)
    # Parameters should be different objects with the same values.
    assert id(model.parameters()[0]) != id(model_copy.parameters()[0])
    assert model.parameters()[0].__dict__ == model_copy.parameters()[0].__dict__


def test_deepcopy_with_nested_modules():
    class CustomClass1(dspy.Module):
        def __init__(self):
            self.lock = threading.Lock()  # Non-copyable object.
            self.cot = dspy.ChainOfThought(dspy.Signature("q -> a"))

    class CustomClass2(dspy.Module):
        def __init__(self):
            self.submodel = CustomClass1()

    model = CustomClass2()
    model_copy = model.deepcopy()
    assert len(model.parameters()) == len(model_copy.parameters())
    # The lock should be refer to the same object (shallow copy).
    assert id(model.submodel.lock) == id(model_copy.submodel.lock)
    # Parameters should be different objects with the same values.
    assert id(model.parameters()[0]) != id(model_copy.parameters()[0])
    assert model.parameters()[0].__dict__ == model_copy.parameters()[0].__dict__


def test_save_and_load_with_json(tmp_path):
    model = dspy.ChainOfThought(dspy.Signature("q -> a"))
    model._predict.signature = model._predict.signature.with_instructions("You are a helpful assistant.")
    model._predict.demos = [
        dspy.Example(q="What is the capital of France?", a="Paris", reasoning="n/a").with_inputs("q", "a")
    ]
    save_path = tmp_path / "model.json"
    model.save(save_path)
    new_model = dspy.ChainOfThought(dspy.Signature("q -> a"))
    new_model.load(save_path)

    assert str(new_model.signature) == str(model.signature)
    assert new_model.demos[0] == model.demos[0].toDict()


def test_save_and_load_with_pkl(tmp_path):
    import datetime

    # `datetime.date` is not json serializable, so we need to save with pickle.
    class MySignature(dspy.Signature):
        """Just a custom signature."""

        current_date: datetime.date = dspy.InputField()
        target_date: datetime.date = dspy.InputField()
        date_diff: int = dspy.OutputField(desc="The difference in days between the current_date and the target_date")

    trainset = [
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 2), "date_diff": 1},
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 3), "date_diff": 2},
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 4), "date_diff": 3},
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 5), "date_diff": 4},
        {"current_date": datetime.date(2024, 1, 1), "target_date": datetime.date(2024, 1, 6), "date_diff": 5},
    ]
    trainset = [dspy.Example(**example).with_inputs("current_date", "target_date") for example in trainset]

    dspy.settings.configure(
        lm=DummyLM([{"date_diff": "1", "reasoning": "n/a"}, {"date_diff": "2", "reasoning": "n/a"}] * 10)
    )

    cot = dspy.ChainOfThought(MySignature)
    cot(current_date=datetime.date(2024, 1, 1), target_date=datetime.date(2024, 1, 2))

    def dummy_metric(example, pred, trace=None):
        return True

    optimizer = dspy.BootstrapFewShot(max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5, metric=dummy_metric)
    compiled_cot = optimizer.compile(cot, trainset=trainset)
    compiled_cot._predict.signature = compiled_cot._predict.signature.with_instructions("You are a helpful assistant.")

    save_path = tmp_path / "model.pkl"
    compiled_cot.save(save_path)

    new_cot = dspy.ChainOfThought(MySignature)
    new_cot.load(save_path)

    assert str(new_cot.signature) == str(compiled_cot.signature)
    assert new_cot.demos == compiled_cot.demos
