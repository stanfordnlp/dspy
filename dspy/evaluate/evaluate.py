import csv
import importlib
import json
import logging
import types
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import pandas as pd

import tqdm

import dspy
from dspy.primitives.program import Program
from dspy.utils.parallelizer import ParallelExecutor

logger = logging.getLogger(__name__)


class Evaluate:
    """Evaluates the performance of a DSPy program on a given devSet and metric.

    Args:
        devset (list[dspy.Example]): The dataset to evaluate on.
        metric (Callable): The metric function to score examples.
        num_threads (int, optional): Number of threads for parallel execution.
        display_progress (bool): Whether to show a progress bar. Defaults to False.
        display_table (int or bool): How many rows of results to display. Defaults to False.
        max_errors (int): Maximum number of errors before aborting. Defaults to 5.
        return_outputs (bool): Whether to return outputs along with scores. Defaults to False.
        provide_traceback (bool): Whether to include tracebacks for errors.
        save_as_json (str, optional): A filepath to save evaluation results.
        timeout (float): Per-task timeout in seconds passed to ParallelExecutor.
            Tasks exceeding this limit are re-submitted up to straggler_limit times.
            Defaults to 120.
        straggler_limit (int): Maximum number of times a timed-out task may be re-submitted
            before it is abandoned. Defaults to 3.
    """

    def __init__(
        self,
        devset,
        metric: Callable = None,
        num_threads: int = 1,
        display_progress: bool = False,
        display_table: int | bool = False,
        max_errors: int = 5,
        return_outputs: bool = False,
        provide_traceback: bool = None,
        save_as_json: str = None,
        timeout: float = 120,
        straggler_limit: int = 3,
        **kwargs,
    ):
        self.devset = devset
        self.metric = metric
        self.num_threads = num_threads
        self.display_progress = display_progress
        self.display_table = display_table
        self.max_errors = max_errors
        self.return_outputs = return_outputs
        self.provide_traceback = provide_traceback
        self.save_as_json = save_as_json
        self.timeout = timeout
        self.straggler_limit = straggler_limit
        if kwargs:
            logger.warning(f"Ignoring unexpected keyword arguments: {list(kwargs.keys())}")

    def __call__(
        self,
        program,
        metric: Callable = None,
        devset = None,
        num_threads = None,
        display_progress = None,
        display_table = None,
        return_outputs = None,
        provide_traceback = None,
    ):
        metric = metric if metric is not None else self.metric
        devset = devset if devset is not None else self.devset
        num_threads = num_threads if num_threads is not None else self.num_threads
        display_progress = display_progress if display_progress is not None else self.display_progress
        display_table = display_table if display_table is not None else self.display_table
        return_outputs = return_outputs if return_outputs is not None else self.return_outputs
        provide_traceback = provide_traceback if provide_traceback is not None else self.provide_traceback

        executor = ParallelExecutor(
            num_threads=num_threads,
            max_errors=self.max_errors,
            disable_progress_bar=not display_progress,
            provide_traceback=provide_traceback,
            compare_results=True,
            timeout=self.timeout,
            straggler_limit=self.straggler_limit,
        )

        def process_example(example, program, metric):
            pred = program(**example.inputs())
            score = metric(example, pred)
            return example, pred, score

        with dspy.context(trace=[]):
            executor_results = executor(process_example, devset, program=program, metric=metric)

        trace = []
        scores = []
        outputs = []

        for example, pred, score in executor_results:
            if pred is not None:
                outputs.append({pred: score})
                scores.append(score)
                trace.append((example, pred, score))
            else:
                outputs.append({})
                scores.append(0)

        avg_metric = sum(scores) / len(devset)

        if display_progress:
            logger.info(f"Average Metric: {avg_metric:.3f} ({int(avg_metric * len(devset))} / {len(devset)})")

        if display_table:
            display_table_size = display_table if isinstance(display_table, int) else len(devset)
            df_all = _convert_trace_to_df(trace[:display_table_size])
            _display_table(df_all)

        if self.save_as_json:
            _save_trace_as_json(trace, self.save_as_json)

        if return_outputs:
            return round(avg_metric * 100, 2), outputs

        return round(avg_metric * 100, 2)


def _convert_trace_to_df(trace):
    import pandas as pd

    rows = []
    for example, pred, score in trace:
        row = {}
        for k, v in example.items():
            row[f"example_{k}"] = str(v)[:100]
        for k, v in pred.items():
            row[f"pred_{k}"] = str(v)[:100]
        row["metric"] = score
        rows.append(row)
    return pd.DataFrame(rows)


def _display_table(df):
    try:
        from IPython.display import display
        display(df)
    except ImportError:
        print(df.to_string())


def _save_trace_as_json(trace, filepath):
    rows = []
    for example, pred, score in trace:
        rows = {
            "example": dict(example.items()),
            "pred": dict(pred.items()),
            "score": score,
        }
        rows.append(row)

    with open(filepath, "w") as f:
        json.dump(rows, f, indent=2, default=str)


# FIXME: TODO: The merge_dicts stuff above is way too quick and dirty.
# TODO: the display_table can't handle False but can handle 0!
# Not sure how it works with True exactly, probably fails too.
