from __future__ import annotations

import contextlib
import logging
import os
import random
import sys
import traceback
import uuid
from collections.abc import Mapping
from typing import Annotated, Any, Protocol, cast

import typer
from datasets import load_dataset  # type: ignore[import-not-found]
from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, StrictStr

import dspy
from dr_dspy.code_eval import extract_dspy_code, run_python_check
from dr_dspy.dspy_event_log import EventLogCallback
from dr_dspy.dspy_programs import (
    count_predictor_demos,
    evaluate_program_with_async_runner,
)
from dr_dspy.event_log import (
    DATABASE_URL_ENV,
    EventStore,
    EventWriter,
    build_event_writer,
)
from dr_dspy.flow import current_flow, event_flow
from dr_dspy.lm_logging import LoggingLM
from dr_dspy.run_metadata import build_run_metadata
from dr_dspy.runtime import configure_multiprocessing
from dspy.signatures.signature import make_signature
from dspy.teleprompt import BootstrapFewShot

# Configuration

DEFAULT_DB_PATH = "./runs.db"
DEFAULT_COMPILED_PATH = "./compiled_humaneval.json"
DEFAULT_EVENT_STORE = EventStore.POSTGRES
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_TRAIN_SIZE = 32
DEFAULT_DEV_SIZE = 32
DEFAULT_NUM_THREADS = 8
DEFAULT_SUBPROCESS_TIMEOUT = 15.0
DEFAULT_SEED = 0
DEFAULT_MAX_BOOTSTRAPPED_DEMOS = 4
DEFAULT_MAX_LABELED_DEMOS = 0
SOLVE_INSTRUCTIONS = (
    "Write a self-contained Python function that satisfies the prompt.\n\n"
    "Include any imports inside the function or at the top. Do not include "
    "tests or example calls. Define exactly the function named in the prompt."
)
MAX_TRACE_SIZE = 10_000
DISPLAY_TABLE_ROWS = 10
ZERO_DEMO_EXIT_CODE = 3


class HarnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_store: EventStore = DEFAULT_EVENT_STORE
    db_path: StrictStr = DEFAULT_DB_PATH
    database_url: StrictStr | None = None
    compiled_path: StrictStr = DEFAULT_COMPILED_PATH
    model: StrictStr = DEFAULT_MODEL
    seed: StrictInt = DEFAULT_SEED
    train_size: StrictInt = DEFAULT_TRAIN_SIZE
    dev_size: StrictInt = DEFAULT_DEV_SIZE
    num_threads: StrictInt = DEFAULT_NUM_THREADS
    timeout: StrictFloat = DEFAULT_SUBPROCESS_TIMEOUT
    max_bootstrapped_demos: StrictInt = DEFAULT_MAX_BOOTSTRAPPED_DEMOS
    max_labeled_demos: StrictInt = DEFAULT_MAX_LABELED_DEMOS


# Signature and Metric

Solve = make_signature(
    {
        "prompt": (str, dspy.InputField()),
        "code": (dspy.Code, dspy.OutputField()),
    },
    instructions=SOLVE_INSTRUCTIONS,
    signature_name="Solve",
)


class HumanEvalMetric:
    def __init__(self, *, writer: EventWriter | None, timeout: float) -> None:
        self._writer = writer
        self._timeout = timeout

    def __call__(
        self, example: dspy.Example, pred: Any, trace: list[Any] | None = None
    ) -> float | bool:
        code = extract_dspy_code(pred)
        result = run_python_check(
            code=code,
            test=example.test,
            entry_point=example.entry_point,
            timeout=self._timeout,
        )
        if self._writer is not None:
            self._writer.put_event(
                "metric.score",
                payload={
                    "task_id": example.task_id,
                    "code": code,
                    "error": result.error,
                },
                example_id=getattr(example, "task_id", None),
                score=result.score,
                error=result.error,
            )
        if trace is None:
            return result.score
        return result.score >= 1.0


# HumanEval Dataset

HumanEvalRow = Mapping[str, Any]


class HumanEvalDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> HumanEvalRow: ...


def build_humaneval_dataset(
    *,
    seed: int = DEFAULT_SEED,
    train_size: int = DEFAULT_TRAIN_SIZE,
    dev_size: int = DEFAULT_DEV_SIZE,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    dataset = cast(
        HumanEvalDataset,
        load_dataset("evalplus/humanevalplus", split="test"),
    )
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    rows = [dataset[i] for i in indices]
    examples = [
        dspy.Example(
            prompt=row["prompt"],
            test=row["test"],
            entry_point=row["entry_point"],
            task_id=row["task_id"],
        ).with_inputs("prompt")
        for row in rows
    ]
    return examples[:train_size], examples[train_size : train_size + dev_size]


# Experiment Flow


def build_humaneval_evaluator(
    devset: list[dspy.Example],
    *,
    metric: HumanEvalMetric,
    num_threads: int,
) -> dspy.Evaluate:
    return dspy.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=num_threads,
        display_progress=True,
        display_table=DISPLAY_TABLE_ROWS,
    )


def log_zero_demo_hint_if_needed(
    compiled: dspy.Module, writer: EventWriter
) -> int:
    counts = count_predictor_demos(compiled)
    if counts.total == 0:
        writer.put_event(
            "bootstrap.no_demos",
            payload={
                "by_predictor": counts.by_predictor,
                "hint": (
                    "BootstrapFewShot produced zero demos at the full-pass "
                    "threshold. Either pass a stronger teacher= to compile() "
                    "or lower the threshold by changing HumanEvalMetric's "
                    "`trace is not None` branch to `return score >= 0.8`."
                ),
            },
        )
    return counts.total


def run_humaneval_bootstrap_flow(
    *,
    config: HarnessConfig,
    writer: EventWriter,
    lm: dspy.BaseLM,
    trainset: list[dspy.Example],
    devset: list[dspy.Example],
) -> int:
    metric = HumanEvalMetric(writer=writer, timeout=config.timeout)
    try:
        dspy.configure(
            lm=lm,
            callbacks=[EventLogCallback(writer)],
            track_usage=True,
            max_trace_size=MAX_TRACE_SIZE,
        )
        writer.put_event(
            "run.start",
            payload=build_run_metadata(config, model_id=lm.model),
        )

        evaluator = build_humaneval_evaluator(
            devset, metric=metric, num_threads=config.num_threads
        )

        with event_flow(writer, "eval_baseline"):
            baseline = evaluate_program_with_async_runner(
                evaluator, dspy.Predict(Solve)
            )

        with event_flow(writer, "optimize"):
            optimizer = BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=config.max_bootstrapped_demos,
                max_labeled_demos=config.max_labeled_demos,
            )
            compiled = optimizer.compile(
                student=dspy.Predict(Solve), trainset=trainset
            )
            total_demos = log_zero_demo_hint_if_needed(compiled, writer)
            if total_demos == 0:
                print(
                    "BootstrapFewShot produced zero demos at the full-pass "
                    "threshold. See bootstrap.no_demos in the event log for "
                    "next steps.",
                    file=sys.stderr,
                )

        with event_flow(writer, "eval_optimized"):
            optimized = evaluate_program_with_async_runner(evaluator, compiled)

        try:
            compiled.save(config.compiled_path)
        except Exception as e:
            print(f"[save compiled] {e!r}", file=sys.stderr)

        writer.put_event(
            "run.end",
            payload={
                "baseline_score": float(baseline.score),
                "optimized_score": float(optimized.score),
                "total_demos": total_demos,
            },
        )
        print(f"baseline:  {baseline.score:.3f}")
        print(f"optimized: {optimized.score:.3f}")
        if total_demos == 0:
            return ZERO_DEMO_EXIT_CODE
        return 0
    except Exception as e:
        traceback.print_exc()
        with contextlib.suppress(Exception):
            writer.put_event(
                "run.error", payload={"error": repr(e)}, error=repr(e)
            )
        return 1


# Real Run Setup


def build_harness_writer(config: HarnessConfig, *, run_id: str) -> EventWriter:
    return build_event_writer(
        event_store=config.event_store,
        run_id=run_id,
        db_path=config.db_path,
        database_url=config.database_url,
        default_flow_fn=current_flow,
    )


def build_real_lm(config: HarnessConfig, writer: EventWriter) -> LoggingLM:
    return LoggingLM(config.model, log=writer.put_event, cache=False)


def run_real_humaneval_bootstrap(config: HarnessConfig) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set; either export it or use the mock "
            "script under scripts/mocks/.",
            file=sys.stderr,
        )
        return 2
    try:
        writer = build_harness_writer(config, run_id=uuid.uuid4().hex)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    try:
        trainset, devset = build_humaneval_dataset(
            seed=config.seed,
            train_size=config.train_size,
            dev_size=config.dev_size,
        )
        return run_humaneval_bootstrap_flow(
            config=config,
            writer=writer,
            lm=build_real_lm(config, writer),
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
    model: Annotated[str, typer.Option("--model")] = DEFAULT_MODEL,
    seed: Annotated[int, typer.Option("--seed")] = DEFAULT_SEED,
    train_size: Annotated[
        int, typer.Option("--train-size")
    ] = DEFAULT_TRAIN_SIZE,
    dev_size: Annotated[int, typer.Option("--dev-size")] = DEFAULT_DEV_SIZE,
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
        model=model,
        seed=seed,
        train_size=train_size,
        dev_size=dev_size,
        num_threads=num_threads,
        timeout=timeout,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )
    raise typer.Exit(run_real_humaneval_bootstrap(config))


if __name__ == "__main__":
    configure_multiprocessing()
    typer.run(main)
