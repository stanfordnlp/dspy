import dspy
from dspy.utils import DummyLM


def test_save_predict(tmp_path):
    predict = dspy.Predict("question->answer")
    predict.save(tmp_path, state_only=False)

    assert (tmp_path / "metadata.json").exists()
    assert (tmp_path / "model.pkl").exists()

    loaded_predict = dspy.load(tmp_path)
    assert isinstance(loaded_predict, dspy.Predict)

    assert predict.signature == loaded_predict.signature


def test_save_custom_model(tmp_path):
    class CustomModel(dspy.Module):
        def __init__(self):
            self.cot1 = dspy.ChainOfThought("question->refined_question")
            self.cot2 = dspy.ChainOfThought("refined_question->answer")

    model = CustomModel()
    model.save(tmp_path, state_only=False)

    loaded_model = dspy.load(tmp_path)
    assert isinstance(loaded_model, CustomModel)

    assert len(model.predictors()) == len(loaded_model.predictors())
    for predictor, loaded_predictor in zip(model.predictors(), loaded_model.predictors()):
        assert predictor.signature == loaded_predictor.signature


def test_save_model_with_custom_signature(tmp_path):
    import datetime

    class MySignature(dspy.Signature):
        """Just a custom signature."""

        current_date: datetime.date = dspy.InputField()
        target_date: datetime.date = dspy.InputField()
        date_diff: int = dspy.OutputField(desc="The difference in days between the current_date and the target_date")

    predict = dspy.Predict(MySignature)
    predict.signature = predict.signature.with_instructions("You are a helpful assistant.")
    predict.save(tmp_path, state_only=False)

    loaded_predict = dspy.load(tmp_path)
    assert isinstance(loaded_predict, dspy.Predict)

    assert predict.signature == loaded_predict.signature


def test_save_compiled_model(tmp_path):
    predict = dspy.Predict("question->answer")
    dspy.settings.configure(lm=DummyLM([{"answer": "blue"}, {"answer": "white"}] * 10))

    trainset = [
        {"question": "What is the color of the sky?", "answer": "blue"},
        {"question": "What is the color of the ocean?", "answer": "blue"},
        {"question": "What is the color of the milk?", "answer": "white"},
        {"question": "What is the color of the coffee?", "answer": "black"},
    ]
    trainset = [dspy.Example(**example).with_inputs("question") for example in trainset]

    def dummy_metric(example, pred, trace=None):
        return True

    optimizer = dspy.BootstrapFewShot(max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5, metric=dummy_metric)
    compiled_predict = optimizer.compile(predict, trainset=trainset)
    compiled_predict.save(tmp_path, state_only=False)

    loaded_predict = dspy.load(tmp_path)
    assert compiled_predict.demos == loaded_predict.demos
    assert compiled_predict.signature == loaded_predict.signature
