"""Exercise real Harbor orchestration with public, in-memory provider extensions."""

import asyncio
from pathlib import Path
from typing import ClassVar

import pytest

pytest.importorskip("harbor")

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.trial.config import EnvironmentConfig, TaskConfig, TrialConfig, VerifierConfig
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

import dspy
from dspy.experimental.harbor import HarborEvaluation
from dspy.utils.dummies import DummyLM


class MemoryEnvironment(BaseEnvironment):
    """Test provider, selected through EnvironmentConfig.import_path, not patched in."""

    instances: ClassVar[list] = []

    @staticmethod
    def type():
        return "memory"

    def _validate_definition(self):
        pass

    async def start(self, force_build):
        self.events = ["start"]
        self.submission = None
        self.instances.append(self)

    async def stop(self, delete):
        self.events.append("stop")

    async def exec(self, command, **kwargs):
        return ExecResult(stdout="", stderr="", return_code=0)

    async def upload_file(self, source_path, target_path):
        self.submission = Path(source_path).read_text()
        self.events.append("submit")

    async def upload_dir(self, source_dir, target_dir):
        pass

    async def download_file(self, source_path, target_path):
        Path(target_path).write_text("")

    async def download_dir(self, source_dir, target_dir):
        Path(target_dir).mkdir(parents=True, exist_ok=True)


class AnswerVerifier(BaseVerifier):
    async def verify(self):
        assert "stop" not in self.environment.events
        self.environment.events.append("verify")
        if self.environment.submission == "verifier-error":
            raise RuntimeError("verifier unavailable")
        return VerifierResult(rewards={"accuracy": float(self.environment.submission == "correct")})


@pytest.fixture
def task(tmp_path):
    MemoryEnvironment.instances.clear()
    path = tmp_path / "task"
    (path / "environment").mkdir(parents=True)
    (path / "tests").mkdir()
    (path / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (path / "instruction.md").write_text("Which answer?")
    (path / "task.toml").write_text('version = "1.0"\n')
    return TrialConfig(
        task=TaskConfig(path=path),
        trials_dir=tmp_path / "trials",
        environment=EnvironmentConfig(import_path=f"{__name__}:MemoryEnvironment"),
        verifier=VerifierConfig(import_path=f"{__name__}:AnswerVerifier"),
    )


async def answer(program, instruction, environment, tools):
    import tempfile

    prediction = await program.acall(question=instruction)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "answer.txt"
        path.write_text(prediction.answer)
        await environment.upload_file(path, "/answer.txt")
    return prediction


def test_real_trial_trace_copy_and_verification(task):
    program = dspy.Predict("question -> answer")
    wrapper = HarborEvaluation(program, answer, reward_key="accuracy")
    candidate = wrapper.deepcopy()
    assert candidate.program is not program
    assert dict(candidate.named_predictors()) == {"program": candidate.program}
    original_config = task.model_dump()
    with dspy.context(lm=DummyLM([{"answer": "correct"}]), trace=[]):
        result = candidate(task=task)
        assert len(dspy.settings.trace) == 1
        assert dspy.settings.trace[0][0] is candidate.program
    assert result.score == 1.0
    assert result.predictions[0].answer == "correct"
    assert result.trial_result.verifier_result.rewards == {"accuracy": 1.0}
    assert MemoryEnvironment.instances[0].events == ["start", "submit", "verify", "stop"]
    assert task.model_dump() == original_config


class Constant(dspy.Module):
    def __init__(self, value):
        self.value = value

    async def aforward(self, question):
        await asyncio.sleep(0.01)
        return dspy.Prediction(answer=self.value)


@pytest.mark.asyncio
async def test_concurrent_candidates_and_repeated_examples_are_isolated(task):
    correct = HarborEvaluation(Constant("correct"), answer, "accuracy")
    wrong = HarborEvaluation(Constant("wrong"), answer, "accuracy")
    results = await asyncio.gather(correct.acall(task=task), wrong.acall(task=task), correct.acall(task=task))
    assert [result.score for result in results] == [1.0, 0.0, 1.0]
    assert len({result.trial_result.trial_name for result in results}) == 3
    assert len(MemoryEnvironment.instances) == 3
    assert all(environment.events[-1] == "stop" for environment in MemoryEnvironment.instances)


def test_callback_failure_is_not_a_zero_reward_and_cleans_up(task):
    async def fail(program, instruction, environment, tools):
        raise ValueError("submission failed")

    wrapper = HarborEvaluation(Constant("correct"), fail, "accuracy")
    with pytest.raises(RuntimeError, match="submission failed"):
        wrapper(task=task)
    assert MemoryEnvironment.instances[0].events[-1] == "stop"
    # A later invocation must not inherit the failed callback's execution state.
    wrapper.run = answer
    assert wrapper(task=task).score == 1.0


def test_missing_reward_is_not_silently_zero(task):
    wrapper = HarborEvaluation(Constant("correct"), answer, "nonexistent")
    with pytest.raises(ValueError, match="nonexistent"):
        wrapper(task=task)
    assert MemoryEnvironment.instances[0].events[-1] == "stop"


def test_verifier_failure_cleans_up(task):
    wrapper = HarborEvaluation(Constant("verifier-error"), answer, "accuracy")
    with pytest.raises(RuntimeError, match="verifier unavailable"):
        wrapper(task=task)
    assert MemoryEnvironment.instances[0].events[-1] == "stop"


def test_harbor_enforces_agent_timeout(task):
    async def wait(program, instruction, environment, tools):
        await asyncio.Event().wait()

    task.agent.override_timeout_sec = 0.01
    wrapper = HarborEvaluation(Constant("correct"), wait, "accuracy")
    with pytest.raises(RuntimeError, match="AgentTimeoutError"):
        wrapper(task=task)
    assert MemoryEnvironment.instances[0].events[-1] == "stop"


def test_threaded_evaluate_runs_fresh_trials(task):
    wrapper = HarborEvaluation(Constant("correct"), answer, "accuracy")
    evaluator = dspy.Evaluate(
        devset=[dspy.Example(task=task).with_inputs("task") for _ in range(2)],
        metric=lambda example, prediction: prediction.score,
        num_threads=2,
        max_errors=1,
    )
    result = evaluator(wrapper)
    assert result.score == 100.0
    assert len({prediction.trial_result.trial_name for _, prediction, _ in result.results}) == 2
    assert len(MemoryEnvironment.instances) == 2
    assert all(
        environment.events == ["start", "submit", "verify", "stop"] for environment in MemoryEnvironment.instances
    )


def test_resume_request_is_not_silently_discarded(task):
    path = task.task.path
    (path / "task.toml").write_text('[[steps]]\nname="one"\n[[steps]]\nname="two"\n')
    for name in ["one", "two"]:
        step = path / "steps" / name
        step.mkdir(parents=True)
        (step / "instruction.md").write_text(name)
    task.agent.resume_trajectory = True
    wrapper = HarborEvaluation(Constant("correct"), answer, "accuracy")
    with pytest.raises(ValueError, match="resume"):
        wrapper(task=task)
    assert not MemoryEnvironment.instances


@pytest.mark.parametrize("mode", ["install_only", "disable_verification"])
def test_unscored_trials_are_not_reported_as_success(task, mode):
    if mode == "install_only":
        task = TrialConfig.model_validate({**task.model_dump(), "install_only": True})
    else:
        task.verifier.disable = True
    wrapper = HarborEvaluation(Constant("correct"), answer, "accuracy")
    with pytest.raises(RuntimeError, match="did not complete"):
        wrapper(task=task)
    assert MemoryEnvironment.instances[0].events[-1] == "stop"


@pytest.mark.asyncio
async def test_cancellation_waits_for_harbor_cleanup(task):
    entered = asyncio.Event()
    caller_loop = asyncio.get_running_loop()

    async def wait(program, instruction, environment, tools):
        caller_loop.call_soon_threadsafe(entered.set)
        await asyncio.Event().wait()

    wrapper = HarborEvaluation(Constant("correct"), wait, "accuracy")
    running = asyncio.create_task(wrapper.acall(task=task))
    await asyncio.wait_for(entered.wait(), timeout=10)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert MemoryEnvironment.instances[0].events[-1] == "stop"
