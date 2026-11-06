"""Run a DSPy child inside Harbor's public custom-agent and trial interfaces.

Requires Harbor (Python 3.12+). Import this module explicitly; Harbor is not a
dependency of the rest of DSPy.
"""

import asyncio
import atexit
import json
import re
import shlex
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from threading import Lock
from uuid import uuid4

from anyio.from_thread import start_blocking_portal
from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

import dspy

_runtime = None
_runtime_lock = Lock()


def _harbor_runtime():
    # Harbor providers share asyncio locks/clients across trials. All candidates
    # must use one persistent loop, including across successive Evaluate calls.
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            manager = start_blocking_portal(name="dspy-harbor")
            portal = manager.__enter__()
            _runtime = portal, portal.call(asyncio.get_running_loop)
            atexit.register(manager.__exit__, None, None, None)
        return _runtime


@dataclass
class _Execution:
    program: dspy.Module
    run: Callable[[dspy.Module, str, BaseEnvironment, list[dspy.Tool]], Awaitable[dspy.Prediction]]
    predictions: list[dspy.Prediction] = field(default_factory=list)


_execution: ContextVar[_Execution] = ContextVar("dspy_harbor_execution")


async def _exec(environment, command, **kwargs):
    result = await environment.exec(command=command, **kwargs)
    if result.return_code != 0:
        raise RuntimeError(
            f"Harbor tool bridge command failed ({result.return_code}): {result.stderr or result.stdout}"
        )
    return result.stdout or ""


async def _mcp_request(environment, directory, message):
    # Files, rather than shell arguments, also handle large tool inputs.
    remote = f"{directory}/{uuid4().hex}.json"
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "request.json"
        path.write_text(json.dumps(message))
        await environment.upload_file(path, remote)
    response = json.loads(await _exec(environment, f"python3 {directory}/client.py request {directory} {remote}"))
    if "error" in response:
        raise RuntimeError(response["error"])
    return response["result"]


@dataclass
class _MCPSession:
    environment: BaseEnvironment
    directory: str
    server: str

    async def call_tool(self, name, arguments):
        from mcp.types import CallToolResult

        result = await _mcp_request(
            self.environment,
            self.directory,
            {"method": "call", "server": self.server, "name": name, "arguments": arguments},
        )
        return CallToolResult.model_validate(result)


@asynccontextmanager
async def _agent_tools(environment, servers, skills_dir):
    tools = []
    if skills_dir:
        files = await _exec(
            environment,
            f"find {shlex.quote(skills_dir)} -mindepth 2 -maxdepth 2 -type f -name SKILL.md -print0",
        )
        skills = {PurePosixPath(path).parent.name: path for path in files.split("\0") if path}
        if skills:

            async def read_skill(name: str):
                path = skills[name]
                return {"path": path, "content": await _exec(environment, f"cat -- {shlex.quote(path)}")}

            tools.append(
                dspy.Tool(
                    read_skill,
                    name="harbor_read_skill",
                    desc="Read a Harbor skill's SKILL.md. Referenced files and scripts remain in the task environment.",
                    args={"name": {"type": "string", "enum": sorted(skills)}},
                )
            )
    if not servers:
        yield tools
        return

    from mcp.types import Tool as MCPTool

    directory = f"/tmp/dspy-mcp-{uuid4().hex}"
    await _exec(environment, f"mkdir -m 700 {directory}")
    try:
        await environment.upload_file(Path(__file__).with_name("_harbor_mcp.py"), f"{directory}/client.py")
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "servers.json"
            path.write_text(json.dumps([server.model_dump(mode="json") for server in servers]))
            await environment.upload_file(path, f"{directory}/servers.json")
        await _exec(environment, f"python3 {directory}/client.py start {directory}")
        definitions = await _mcp_request(environment, directory, {"method": "list"})
        names = set()
        for definition in definitions:
            session = _MCPSession(environment, directory, definition["server"])
            tool = dspy.Tool.from_mcp_tool(
                session, MCPTool.model_validate(definition["tool"]), result_mode="structured"
            )
            tool.name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"mcp__{session.server}__{tool.name}")
            if tool.name in names:
                raise ValueError(f"MCP tool names collide after namespacing: {tool.name}")
            names.add(tool.name)
            tools.append(tool)
        yield tools
    finally:
        # Close sessions even if the callback fails or Harbor cancels its timeout.
        # Leave diagnostics in a retained environment if shutdown itself fails.
        await _exec(
            environment,
            f"if test -f {directory}/client.py; then python3 {directory}/client.py stop {directory}; fi",
            timeout_sec=30,
        )
        await _exec(environment, f"rm -rf {directory}")


class DSPyAgent(BaseAgent):
    """Harbor's importable agent bridge; used only by HarborEvaluation in-process."""

    capabilities = AgentCapabilities(mcp_servers=True, skills=True)

    @staticmethod
    def name() -> str:
        return "dspy-evaluation"

    def version(self) -> str:
        return dspy.__version__

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        execution = _execution.get()
        async with _agent_tools(environment, self.mcp_servers, self.skills_dir) as tools:
            prediction = await execution.run(execution.program, instruction, environment, tools)
        execution.predictions.append(prediction)


class HarborEvaluation(dspy.Module):
    """Evaluate an optimizable child through a complete Harbor trial.

    Args:
        program: The child to optimize and subsequently extract from this wrapper.
        run: Async callback ``(program, instruction, environment, tools) -> Prediction``.
            Tools contain the task's MCP tools and a skill reader, when configured.
            It binds task-specific tools/inputs, calls the supplied candidate, and
            submits its output through the environment's public interface. It must
            not capture another program or pass verifier data into the child.
        reward_key: The named Harbor reward exposed as ``prediction.score``.

    ``forward(task=TrialConfig(...))`` copies the configuration, replaces its agent
    with the DSPy bridge, and assigns a fresh trial name. Harbor owns setup,
    verification, artifacts, and teardown. Only configuration and the callback are
    stored on the module; live resources are scoped to each invocation.

    The result contains ``score``, ``rewards``, ``predictions`` (one per agent
    phase), and the complete Harbor ``trial_result``. Metrics should read these
    results without invoking the verifier again. Harbor-reported errors raise
    rather than being silently converted to a legitimate zero reward.
    """

    def __init__(
        self,
        program: dspy.Module,
        run: Callable[[dspy.Module, str, BaseEnvironment, list[dspy.Tool]], Awaitable[dspy.Prediction]],
        reward_key: str = "reward",
    ):
        super().__init__()
        self.program = program
        self.run = run
        self.reward_key = reward_key

    def forward(self, task: TrialConfig) -> dspy.Prediction:
        portal, _ = _harbor_runtime()
        return portal.call(self._run_trial, task)

    async def aforward(self, task: TrialConfig) -> dspy.Prediction:
        portal, loop = _harbor_runtime()
        if asyncio.get_running_loop() is loop:
            return await self._run_trial(task)

        started = Future()

        async def run():
            started.set_result(asyncio.current_task())
            return await self._run_trial(task)

        # The portal copies each caller's context, including the DSPy trace list.
        pending = asyncio.wrap_future(portal.start_task_soon(run))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            # Cancelling the concurrent future would report completion before
            # Harbor's finally blocks finish. Cancel its task and await cleanup.
            running = await asyncio.shield(asyncio.wrap_future(started))
            loop.call_soon_threadsafe(running.cancel)
            try:
                await asyncio.shield(pending)
            except asyncio.CancelledError:
                pass
            raise

    async def _run_trial(self, task: TrialConfig) -> dspy.Prediction:
        config = task.model_copy(deep=True)
        # Replace the agent implementation, not Harbor's execution policy. In
        # particular, retain env/timeouts and let Harbor reject unsupported resume.
        config.agent.name = None
        config.agent.import_path = "dspy.experimental.harbor:DSPyAgent"
        config.agent.model_name = None
        config.agent.kwargs = {}
        config.trial_name = f"dspy-{uuid4().hex}"
        execution = _Execution(self.program, self.run)
        token = _execution.set(execution)
        try:
            trial = await Trial.create(config)
            result = await trial.run()
        finally:
            _execution.reset(token)

        errors = [result.exception_info] + [step.exception_info for step in result.step_results or []]
        for error in errors:
            if error is not None:
                raise RuntimeError(
                    f"Harbor trial {result.trial_name} failed: {error.exception_type}: {error.exception_message}"
                )
        if not execution.predictions or result.verifier_result is None:
            raise RuntimeError(f"Harbor trial {result.trial_name} did not complete agent execution and verification")
        rewards = result.verifier_result.rewards
        if rewards is None or self.reward_key not in rewards:
            raise ValueError(f"Harbor trial did not return reward {self.reward_key!r}: {rewards}")
        return dspy.Prediction(
            score=rewards[self.reward_key],
            rewards=rewards,
            predictions=execution.predictions,
            trial_result=result,
        )
