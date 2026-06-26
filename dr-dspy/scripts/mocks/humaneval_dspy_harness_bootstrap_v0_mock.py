from __future__ import annotations

import logging
import re
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

import dspy
from dr_dspy.event_log import DATABASE_URL_ENV, EventStore
from dr_dspy.lm_logging import LoggingCallableLM
from dr_dspy.runtime import configure_multiprocessing

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from humaneval_dspy_harness_bootstrap_v0 import (  # noqa: E402
    DEFAULT_COMPILED_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_EVENT_STORE,
    DEFAULT_MAX_BOOTSTRAPPED_DEMOS,
    DEFAULT_MAX_LABELED_DEMOS,
    DEFAULT_NUM_THREADS,
    DEFAULT_SUBPROCESS_TIMEOUT,
    HarnessConfig,
    build_harness_writer,
    run_humaneval_bootstrap_flow,
)

MOCK_MODEL = "callable/mock"
MockSolver = Callable[[list[dict[str, Any]], dict[str, Any]], dict[str, str]]


# Mock Fixtures


def make_mock_dataset() -> tuple[list[dspy.Example], list[dspy.Example]]:
    rows = [
        {
            "task_id": "mock/add",
            "prompt": "def add(a, b):\n    'return a + b'\n",
            "entry_point": "add",
            "test": (
                "def check(candidate):\n"
                "    assert candidate(1,2)==3\n"
                "    assert candidate(-1,1)==0\n"
            ),
        },
        {
            "task_id": "mock/sub",
            "prompt": "def sub(a, b):\n    'return a - b'\n",
            "entry_point": "sub",
            "test": (
                "def check(candidate):\n"
                "    assert candidate(3,2)==1\n"
                "    assert candidate(0,5)==-5\n"
            ),
        },
        {
            "task_id": "mock/mul",
            "prompt": "def mul(a, b):\n    'return a * b'\n",
            "entry_point": "mul",
            "test": "def check(candidate):\n    assert candidate(2,3)==6\n",
        },
        {
            "task_id": "mock/identity",
            "prompt": "def identity(x):\n    'return x'\n",
            "entry_point": "identity",
            "test": "def check(candidate):\n    assert candidate(7)==7\n",
        },
    ]
    train = [dspy.Example(**row).with_inputs("prompt") for row in rows]
    dev = [dspy.Example(**row).with_inputs("prompt") for row in rows]
    return train, dev


def mock_solver(
    messages: list[dict[str, Any]], _kwargs: dict[str, Any]
) -> dict[str, str]:
    text = "\n".join(str(message.get("content", "")) for message in messages)
    body_map = {
        "add": "def add(a, b):\n    return a + b\n",
        "sub": "def sub(a, b):\n    return a - b\n",
        "mul": "def mul(a, b):\n    return a * b\n",
        "identity": "def identity(x):\n    return x\n",
    }
    matches = re.findall(r"\bdef\s+(add|sub|mul|identity)\s*\(", text)
    if matches:
        return {"code": body_map[matches[-1]]}
    return {"code": "def f():\n    return None\n"}


# Mock Run Setup


def build_mock_lm(
    solver: MockSolver,
    *,
    log: Callable[..., None],
) -> LoggingCallableLM:
    return LoggingCallableLM(solver, log=log, model=MOCK_MODEL)


def run_mock_humaneval_bootstrap(config: HarnessConfig) -> int:
    try:
        writer = build_harness_writer(config, run_id=uuid.uuid4().hex)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    try:
        trainset, devset = make_mock_dataset()
        return run_humaneval_bootstrap_flow(
            config=config,
            writer=writer,
            lm=build_mock_lm(mock_solver, log=writer.put_event),
            trainset=trainset,
            devset=devset,
        )
    finally:
        writer.close()


# CLI


def main(
    event_store: Annotated[
        EventStore,
        typer.Option("--event-store", help="Event log backend to use."),
    ] = DEFAULT_EVENT_STORE,
    db_path: Annotated[str, typer.Option("--db-path")] = DEFAULT_DB_PATH,
    database_url: Annotated[
        str | None,
        typer.Option(
            "--database-url",
            help=(
                "Postgres URL for --event-store postgres; defaults to "
                f"{DATABASE_URL_ENV}."
            ),
        ),
    ] = None,
    compiled_path: Annotated[
        str, typer.Option("--compiled-path")
    ] = DEFAULT_COMPILED_PATH,
    num_threads: Annotated[
        int, typer.Option("--num-threads")
    ] = DEFAULT_NUM_THREADS,
    timeout: Annotated[
        float, typer.Option("--timeout")
    ] = DEFAULT_SUBPROCESS_TIMEOUT,
    max_bootstrapped_demos: Annotated[
        int, typer.Option("--max-bootstrapped-demos")
    ] = DEFAULT_MAX_BOOTSTRAPPED_DEMOS,
    max_labeled_demos: Annotated[
        int, typer.Option("--max-labeled-demos")
    ] = DEFAULT_MAX_LABELED_DEMOS,
) -> None:
    logging.getLogger("dspy").setLevel(logging.WARNING)
    config = HarnessConfig(
        event_store=event_store,
        db_path=db_path,
        database_url=database_url,
        compiled_path=compiled_path,
        model=MOCK_MODEL,
        seed=0,
        train_size=4,
        dev_size=4,
        num_threads=num_threads,
        timeout=timeout,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )
    raise typer.Exit(run_mock_humaneval_bootstrap(config))


if __name__ == "__main__":
    configure_multiprocessing()
    typer.run(main)
