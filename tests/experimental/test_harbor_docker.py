"""Opt-in real Docker trials: HARBOR_DOCKER_TESTS=1 pytest <this file>.

Uses Harbor's Docker provider and built-in verifier, without mocked orchestration.
Requires a working Docker daemon and the ubuntu:24.04 image (pulled if missing).
"""

import asyncio
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import pytest

pytest.importorskip("harbor")

from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.task.config import MCPServerConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig

import dspy
from dspy.experimental.harbor import HarborEvaluation
from dspy.utils.dummies import DummyLM

pytestmark = pytest.mark.skipif(
    os.environ.get("HARBOR_DOCKER_TESTS") != "1", reason="Set HARBOR_DOCKER_TESTS=1 to run Docker trials"
)


class ToolAnswer(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict("question, evidence -> answer")

    async def aforward(self, question, inspect_environment):
        evidence = await inspect_environment()
        return await self.predict.acall(question=question, evidence=evidence)


async def submit(program, instruction, environment, tools):
    async def inspect_environment():
        result = await environment.exec(command='printf "%s/%s" "$BASE_SETTING" "$AGENT_SETTING"')
        assert result.return_code == 0
        return result.stdout

    prediction = await program.acall(question=instruction, inspect_environment=inspect_environment)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "answer.txt"
        path.write_text(prediction.answer)
        await environment.upload_file(path, "/app/answer.txt")
    # This file must NOT be transferred to a separate verifier environment.
    result = await environment.exec(command="touch /app/agent-only")
    assert result.return_code == 0
    return prediction


@pytest.fixture
def docker_task(tmp_path):
    path = tmp_path / "task"
    (path / "environment").mkdir(parents=True)
    (path / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /app\n")
    (path / "instruction.md").write_text("Answer the question using the environment tool.")
    return TrialConfig(
        task=TaskConfig(path=path),
        trials_dir=tmp_path / "trials",
        environment=EnvironmentConfig(type="docker", delete=True),
        agent=AgentConfig(env={"AGENT_SETTING": "agent-value"}),
    )


def write_task(config, *, mode="shared", multi=False, strategy="mean", bundled=False, reward_json=False):
    path = config.task.path
    header = 'artifacts = [{source="/app/answer.txt", destination="answer.txt"}]\n'
    header += f'multi_step_reward_strategy = "{strategy}"\n'
    header += '[environment.env]\nBASE_SETTING = "base-value"\n'
    header += f'[verifier]\nenvironment_mode = "{mode}"\n'
    header += '[verifier.env]\nJUDGE_SETTING = "judge-value"\n'
    if multi:
        header += '\n[[steps]]\nname="first"\n\n[[steps]]\nname="second"\n'
    (path / "task.toml").write_text(header)
    for name in ["first", "second"] if multi else [None]:
        directory = path / "steps" / name if name else path
        (directory / "tests").mkdir(parents=True)
        if name:
            (directory / "instruction.md").write_text(f"Answer for {name}.")
        # An asymmetric pair detects accidental final-only or mean-only handling.
        separate_check = "! test -e /app/agent-only" if mode == "separate" else "test -e /app/agent-only"
        script = f"""#!/bin/bash
set -eu
test "$JUDGE_SETTING" = judge-value
test -z "${{AGENT_SETTING:-}}"
{separate_check}
score=0
if [ "$(cat /app/answer.txt)" = correct ]; then score=1; fi
mkdir -p /logs/verifier
"""
        if reward_json:
            script += 'printf \'{"accuracy": %s, "diagnostic": 0.25}\\n\' "$score" > /logs/verifier/reward.json\n'
        else:
            script += 'printf "%s\\n" "$score" > /logs/verifier/reward.txt\n'
        (directory / "tests" / "test.sh").write_text(script)
        if bundled:
            (directory / "tests" / "Dockerfile").write_text("FROM ubuntu:24.04\nCOPY test.sh /tests/test.sh\n")


@pytest.mark.parametrize(
    "mode,multi,strategy,bundled,reward_json,expected",
    [
        pytest.param("shared", False, "mean", False, False, 1.0, id="single-shared-text"),
        pytest.param("shared", False, "mean", False, True, 1.0, id="single-named-json-rewards"),
        pytest.param("separate", False, "mean", False, False, 1.0, id="separate-artifact-transfer"),
        pytest.param("separate", False, "mean", True, False, 1.0, id="separate-bundled-verifier-image"),
        pytest.param("shared", True, "mean", False, False, 0.5, id="multi-shared-mean"),
        pytest.param("shared", True, "final", False, False, 1.0, id="multi-shared-final"),
        pytest.param("separate", True, "mean", False, False, 0.5, id="multi-separate-mean"),
    ],
)
def test_public_docker_configurations(docker_task, mode, multi, strategy, bundled, reward_json, expected):
    write_task(docker_task, mode=mode, multi=multi, strategy=strategy, bundled=bundled, reward_json=reward_json)
    wrapper = HarborEvaluation(ToolAnswer(), submit, "accuracy" if reward_json else "reward")
    replies = [{"answer": "wrong"}, {"answer": "correct"}] if multi else [{"answer": "correct"}]
    original = docker_task.model_dump()
    with dspy.context(lm=DummyLM(replies), trace=[]):
        result = wrapper(task=docker_task)
        traces = dspy.settings.trace
        assert len(traces) == len(replies)
        assert all(trace[0] is wrapper.program.predict for trace in traces)
        assert all(trace[1]["evidence"] == "base-value/agent-value" for trace in traces)
    assert result.score == expected
    assert [prediction.answer for prediction in result.predictions] == [reply["answer"] for reply in replies]
    assert docker_task.model_dump() == original
    trial_dir = docker_task.trials_dir / result.trial_result.trial_name
    assert (trial_dir / "result.json").exists()
    if not multi:
        assert (trial_dir / "artifacts" / "answer.txt").read_text() == "correct"
    else:
        assert [step.verifier_result.rewards["reward"] for step in result.trial_result.step_results] == [0.0, 1.0]
    if reward_json:
        assert result.rewards == {"accuracy": 1.0, "diagnostic": 0.25}


class InspectCleanupDocker(DockerEnvironment):
    """Observe a real container before teardown, including failed agent runs."""

    observations: ClassVar[list] = []

    async def stop(self, delete):
        try:
            result = await self.exec(command="python3 /app/check-cleanup.py")
            self.observations.append(result)
        finally:
            await super().stop(delete)


@pytest.fixture
def mcp_task(docker_task, tmp_path):
    path = docker_task.task.path
    (path / "environment" / "Dockerfile").write_text(
        "FROM ubuntu:24.04\n"
        "RUN apt-get update && apt-get install -y python3 python3-venv && "
        "python3 -m venv /opt/mcp && /opt/mcp/bin/pip install 'mcp==1.30.0'\n"
        "ENV PATH=/opt/mcp/bin:$PATH\nWORKDIR /app\n"
        "COPY . /app/\nRUN printf sandbox-only > /app/sandbox-marker\n"
    )
    shutil.copyfile(Path(__file__).parent / "resources" / "harbor_mcp_server.py", path / "environment" / "server.py")
    # Check live processes, not merely socket deletion, before Docker stops them.
    (path / "environment" / "check-cleanup.py").write_text("""from pathlib import Path
assert not list(Path("/tmp").glob("dspy-mcp-*")), "MCP bridge files leaked"
for file in Path("/proc").glob("[0-9]*/cmdline"):
    try:
        args = file.read_bytes().split(b"\\0")
    except FileNotFoundError:
        continue
    assert not (b"/app/server.py" in args and b"stdio" in args), "stdio server leaked"
    assert b"import time; time.sleep(300)" not in args, "initializing server leaked"
    assert not any(arg.startswith(b"/tmp/dspy-mcp-") for arg in args), "MCP client leaked"
print("sessions closed before environment teardown")
""")
    runtime_skill = tmp_path / "runtime-skill"
    runtime_skill.mkdir()
    (runtime_skill / "SKILL.md").write_text(
        "---\nname: runtime-skill\ndescription: Runtime advice\n---\nCount carefully."
    )
    docker_task.agent.skills = [str(runtime_skill)]
    docker_task.environment.import_path = f"{__name__}:InspectCleanupDocker"
    InspectCleanupDocker.observations.clear()
    return docker_task


def configure_mcp_task(config, transport):
    path = config.task.path
    text = "[environment]\n"
    # The runtime config must override the task's server with the same name.
    if transport == "stdio":
        text += '[[environment.mcp_servers]]\nname="counter"\ntransport="stdio"\n'
        text += 'command="does-not-exist"\n'
        config.agent.mcp_servers = [
            MCPServerConfig(name=name, transport="stdio", command="python3", args=["/app/server.py", "stdio", name])
            for name in ["counter", "other"]
        ]
    else:
        endpoint = "sse" if transport == "sse" else "mcp"
        text += '[[environment.mcp_servers]]\nname="counter"\n'
        text += f'transport="{transport}"\nurl="http://mcp-server:8123/{endpoint}"\n'
        (path / "environment" / "docker-compose.yaml").write_text(f"""services:
  main:
    depends_on:
      mcp-server:
        condition: service_healthy
  mcp-server:
    build: .
    command: ["python3", "/app/server.py", "{transport}", "counter"]
    healthcheck:
      test: ["CMD", "python3", "-c", "import socket; socket.create_connection(('localhost',8123),2).close()"]
      interval: 1s
      timeout: 3s
      retries: 30
""")
    (path / "task.toml").write_text(text)
    (path / "tests").mkdir()
    (path / "tests" / "test.sh").write_text(
        "#!/bin/bash\nset -eu\npython3 /app/check-cleanup.py\n"
        'test "$(cat /app/answer.txt)" = correct\nmkdir -p /logs/verifier\necho 1 > /logs/verifier/reward.txt\n'
    )


@pytest.mark.parametrize("transport", ["stdio", "sse", "streamable-http"])
def test_sandbox_mcp_and_runtime_skills(mcp_task, transport):
    configure_mcp_task(mcp_task, transport)

    async def run(program, instruction, environment, tools):
        tools = {tool.name: tool for tool in tools}
        counter = tools["mcp__counter__counter"]
        first = await counter.acall(step=3)
        second = await counter.acall(step=-1, payload="λ" * 150_000)
        assert first["count"] == 3
        assert second == {**first, "count": 2, "payload": "λ" * 150_000}
        assert first["sandbox"] == "sandbox-only"
        if transport == "stdio":
            assert first["setting"] == "agent-value"
            assert (await tools["mcp__other__counter"].acall(step=7))["count"] == 7
        with pytest.raises(RuntimeError, match="intentional MCP failure"):
            await tools["mcp__counter__fail"].acall()
        skill = await tools["harbor_read_skill"].acall(name="runtime-skill")
        assert skill["content"].endswith("Count carefully.")
        assert skill["path"].endswith("/runtime-skill/SKILL.md")
        with pytest.raises(ValueError):
            await tools["harbor_read_skill"].acall(name="../../etc/passwd")

        async def evidence():
            return json.dumps(await counter.acall(step=0))

        prediction = await program.acall(question=instruction, inspect_environment=evidence)
        result = await environment.exec(command=f"printf %s {prediction.answer} > /app/answer.txt")
        assert result.return_code == 0
        return prediction

    wrapper = HarborEvaluation(ToolAnswer(), run)
    with dspy.context(lm=DummyLM([{"answer": "correct"}]), trace=[]):
        assert wrapper(task=mcp_task).score == 1.0
        assert dspy.settings.trace[0][0] is wrapper.program.predict
        assert json.loads(dspy.settings.trace[0][1]["evidence"])["count"] == 2
    assert len(InspectCleanupDocker.observations) == 1
    assert InspectCleanupDocker.observations[0].return_code == 0


@pytest.mark.parametrize("failure", ["startup", "startup-timeout", "callback", "cancel-tool", "agent-timeout"])
def test_mcp_cleanup_on_failure(mcp_task, failure):
    configure_mcp_task(mcp_task, "stdio")
    if failure == "startup":
        mcp_task.agent.mcp_servers[1].command = "missing-mcp-command"
    if failure == "startup-timeout":
        mcp_task.agent.mcp_servers[1].args = ["-c", "import time; time.sleep(300)"]
    if failure in {"agent-timeout", "startup-timeout"}:
        mcp_task.agent.override_timeout_sec = 10
    entered = False

    async def run(program, instruction, environment, tools):
        nonlocal entered
        entered = True
        if failure == "callback":
            raise ValueError("callback failed after MCP initialization")
        tool = next(tool for tool in tools if tool.name == "mcp__counter__wait")
        await asyncio.wait_for(tool.acall(), timeout=1 if failure == "cancel-tool" else 300)

    with pytest.raises(RuntimeError, match=r"MCP client failed|callback failed|TimeoutError"):
        HarborEvaluation(ToolAnswer(), run)(task=mcp_task)
    assert entered == (failure not in {"startup", "startup-timeout"})
    assert len(InspectCleanupDocker.observations) == 1
    assert InspectCleanupDocker.observations[0].return_code == 0


def test_task_bundled_skill_without_mcp_dependencies(docker_task):
    write_task(docker_task)
    path = docker_task.task.path
    config = (path / "task.toml").read_text()
    (path / "task.toml").write_text(
        config.replace("[environment.env]", '[environment]\nskills_dir="/opt/skills with spaces"\n[environment.env]')
    )
    skill = path / "environment" / "skills" / "bundled"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("Read ./reference.txt before answering.")
    (skill / "reference.txt").write_text("Supporting material stays in the environment.")
    (path / "environment" / "Dockerfile").write_text(
        'FROM ubuntu:24.04\nWORKDIR /app\nCOPY ["skills", "/opt/skills with spaces"]\n'
    )

    async def run(program, instruction, environment, tools):
        assert len(tools) == 1
        blob = await tools[0].acall(name="bundled")
        assert blob == {
            "path": "/opt/skills with spaces/bundled/SKILL.md",
            "content": "Read ./reference.txt before answering.",
        }
        return await submit(program, instruction, environment, tools)

    with dspy.context(lm=DummyLM([{"answer": "correct"}])):
        assert HarborEvaluation(ToolAnswer(), run)(task=docker_task).score == 1.0


@pytest.mark.parametrize("multi", [False, True], ids=["single-step", "multi-step"])
def test_gepa_optimizes_and_extracts_child(docker_task, multi):
    class InstructionLM(DummyLM):
        """Follow the candidate instruction, independently of call order."""

        def __init__(self):
            super().__init__([])

        def __call__(self, prompt=None, messages=None, **kwargs):
            value = "correct" if "Return exactly correct." in messages[0]["content"] else "wrong"
            return [self._format_answer_fields({"answer": value})]

        async def acall(self, prompt=None, messages=None, **kwargs):
            return self(prompt=prompt, messages=messages, **kwargs)

    def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
        return dspy.Prediction(score=prediction.score, feedback=f"Harbor reward: {prediction.score}")

    def propose(candidate, reflective_dataset, components_to_update):
        records = reflective_dataset["program.predict"]
        assert records
        assert records[0]["Inputs"]["evidence"] == "base-value/agent-value"
        assert records[0]["Generated Outputs"]["answer"] == "wrong"
        assert "Harbor reward: 0.0" in records[0]["Feedback"]
        return {"program.predict": "Return exactly correct."}

    write_task(docker_task, multi=multi)
    wrapper = HarborEvaluation(ToolAnswer(), submit)
    wrapper.program.predict.signature = wrapper.program.predict.signature.with_instructions("Return exactly wrong.")
    examples = [dspy.Example(task=docker_task).with_inputs("task") for _ in range(4)]
    evaluator = dspy.Evaluate(devset=examples[2:], metric=metric, num_threads=2, max_errors=1)
    with dspy.context(lm=InstructionLM()):
        assert evaluator(wrapper).score == 0.0
        compiled = dspy.GEPA(
            metric=metric,
            instruction_proposer=propose,
            max_metric_calls=6,
            reflection_minibatch_size=2,
            num_threads=2,
            use_merge=False,
        ).compile(wrapper, trainset=examples[:2], valset=examples[2:])
        assert evaluator(compiled).score == 100.0
        assert evaluator(wrapper).score == 0.0

        # Extraction and state-only serialization must not need a live trial.
        saved = docker_task.trials_dir / "child.json"
        compiled.program.save(saved)
        restored = ToolAnswer()
        restored.load(saved)

        async def evidence():
            return "independent environment"

        for child in (compiled.program, restored):
            result = asyncio.run(child.acall(question="New question", inspect_environment=evidence))
            assert result.answer == "correct"
