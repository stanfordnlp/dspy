import asyncio

import pytest

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match
from dspy.predict import Predict
from dspy.utils.dummies import DummyLM


def new_example(question: str, answer: str):
    return dspy.Example(question=question, answer=answer).with_inputs("question")


class AsyncPredictProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict("question -> answer")

    async def aforward(self, question: str):
        return await self.predict.acall(question=question)


class SyncOnlyProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = Predict("question -> answer")

    def forward(self, question: str):
        return self.predict(question=question)


class FlakyAsyncProgram(dspy.Module):
    async def aforward(self, question: str):
        if question == "boom":
            raise ValueError("boom")
        return dspy.Prediction(answer="ok")


@pytest.mark.anyio
async def test_evaluate_acall_async_program_sync_metric():
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    evaluator = Evaluate(devset=devset, metric=answer_exact_match, num_threads=2, display_progress=False)
    with dspy.context(lm=DummyLM({"What is 1+1?": {"answer": "2"}, "What is 2+2?": {"answer": "4"}})):
        result = await evaluator.acall(AsyncPredictProgram())

    assert result.score == 100.0


@pytest.mark.anyio
async def test_evaluate_acall_sync_program_async_metric():
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict("question -> answer")

    async def async_metric(example, pred):
        await asyncio.sleep(0)
        return answer_exact_match(example, pred)

    evaluator = Evaluate(devset=devset, metric=async_metric, num_threads=2, display_progress=False)
    with dspy.context(lm=DummyLM({"What is 1+1?": {"answer": "2"}, "What is 2+2?": {"answer": "4"}})):
        result = await evaluator.acall(program)

    assert result.score == 100.0


@pytest.mark.anyio
async def test_evaluate_acall_uses_failure_score_on_exceptions():
    devset = [new_example("ok", "ok"), new_example("boom", "ok")]

    def metric(example, pred):
        return float(example.answer == pred.answer)

    evaluator = Evaluate(
        devset=devset,
        metric=metric,
        num_threads=2,
        display_progress=False,
        failure_score=0.25,
        max_errors=10,
    )

    result = await evaluator.acall(FlakyAsyncProgram())

    assert [row[2] for row in result.results] == [1.0, 0.25]
    assert result.score == 62.5


@pytest.mark.anyio
async def test_evaluate_acall_sync_only_program():
    """Sync-only programs (no aforward) are automatically wrapped via asyncify."""
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    evaluator = Evaluate(devset=devset, metric=answer_exact_match, num_threads=2, display_progress=False)
    with dspy.context(lm=DummyLM({"What is 1+1?": {"answer": "2"}, "What is 2+2?": {"answer": "4"}})):
        result = await evaluator.acall(SyncOnlyProgram())

    assert result.score == 100.0


@pytest.mark.anyio
async def test_evaluate_acall_max_errors_raises():
    """When max_errors is exceeded, acall raises RuntimeError."""
    devset = [new_example("boom", "ok")] * 3

    def metric(example, pred):
        return 0.0

    evaluator = Evaluate(
        devset=devset,
        metric=metric,
        num_threads=1,
        display_progress=False,
        max_errors=2,
    )

    with pytest.raises(RuntimeError, match="Execution cancelled"):
        await evaluator.acall(FlakyAsyncProgram())
