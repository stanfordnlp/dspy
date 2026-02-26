import asyncio
import csv
import importlib
import inspect
import json
import logging
import sys
import traceback
import types
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import pandas as pd

import tqdm

import dspy
from dspy.primitives.prediction import Prediction
from dspy.utils.callback import with_callbacks
from dspy.utils.parallelizer import ParallelExecutor

try:
    from IPython.display import HTML
    from IPython.display import display as display

except ImportError:

    def display(obj: Any):
        """
        Display the specified Python object in the console.

        :param obj: The Python object to display.
        """
        print(obj)

    def HTML(x: str) -> str:  # noqa: N802
        """
        Obtain the HTML representation of the specified string.
        """
        # NB: This method exists purely for code compatibility with the IPython HTML() function in
        # environments where IPython is not available. In such environments where IPython is not
        # available, this method will simply return the input string.
        return x


# TODO: Counting failures and having a max_failure count. When that is exceeded (also just at the end),
# we print the number of failures, the first N examples that failed, and the first N exceptions raised.

logger = logging.getLogger(__name__)


class EvaluationResult(Prediction):
    """
    A class that represents the result of an evaluation.
    It is a subclass of `dspy.Prediction` that contains the following fields

    - score: An float value (e.g., 67.30) representing the overall performance
    - results: a list of (example, prediction, score) tuples for each example in devset
    """

    def __init__(self, score: float, results: list[tuple["dspy.Example", "dspy.Example", Any]]):
        super().__init__(score=score, results=results)

    def __repr__(self):
        return f"EvaluationResult(score={self.score}, results=<list of {len(self.results)} results>)"


class Evaluate:
    """DSPy Evaluate class.

    This class is used to evaluate the performance of a DSPy program. Users need to provide a evaluation dataset and
    a metric function in order to use this class. This class supports parallel evaluation on the provided dataset.
    """

    def __init__(
        self,
        *,
        devset: list["dspy.Example"],
        metric: Callable | None = None,
        num_threads: int | None = None,
        display_progress: bool = False,
        display_table: bool | int = False,
        max_errors: int | None = None,
        provide_traceback: bool | None = None,
        failure_score: float = 0.0,
        save_as_csv: str | None = None,
        save_as_json: str | None = None,
        **kwargs,
    ):
        """
        Args:
            devset (list[dspy.Example]): the evaluation dataset.
            metric (Callable): The metric function to use for evaluation.
            num_threads (Optional[int]): The number of threads to use for parallel evaluation.
            display_progress (bool): Whether to display progress during evaluation.
            display_table (Union[bool, int]): Whether to display the evaluation results in a table.
                If a number is passed, the evaluation results will be truncated to that number before displayed.
            max_errors (Optional[int]): The maximum number of errors to allow before
                stopping evaluation. If ``None``, inherits from ``dspy.settings.max_errors``.
            provide_traceback (Optional[bool]): Whether to provide traceback information during evaluation.
            failure_score (float): The default score to use if evaluation fails due to an exception.
            save_as_csv (Optional[str]): The file name where the csv will be saved.
            save_as_json (Optional[str]): The file name where the json will be saved.

        """
        self.devset = devset
        self.metric = metric
        self.num_threads = num_threads
        self.display_progress = display_progress
        self.display_table = display_table
        self.max_errors = max_errors
        self.provide_traceback = provide_traceback
        self.failure_score = failure_score
        self.save_as_csv = save_as_csv
        self.save_as_json = save_as_json

        if "return_outputs" in kwargs:
            raise ValueError("`return_outputs` is no longer supported. Results are always returned inside the `results` field of the `EvaluationResult` object.")

    @with_callbacks
    def __call__(
        self,
        program: "dspy.Module",
        metric: Callable | None = None,
        devset: list["dspy.Example"] | None = None,
        num_threads: int | None = None,
        display_progress: bool | None = None,
        display_table: bool | int | None = None,
        callback_metadata: dict[str, Any] | None = None,
        save_as_csv: str | None = None,
        save_as_json: str | None = None,
    ) -> EvaluationResult:
        """
        Args:
            program (dspy.Module): The DSPy program to evaluate.
            metric (Callable): The metric function to use for evaluation. if not provided, use `self.metric`.
            devset (list[dspy.Example]): the evaluation dataset. if not provided, use `self.devset`.
            num_threads (Optional[int]): The number of threads to use for parallel evaluation. if not provided, use
                `self.num_threads`.
            display_progress (bool): Whether to display progress during evaluation. if not provided, use
                `self.display_progress`.
            display_table (Union[bool, int]): Whether to display the evaluation results in a table. if not provided, use
                `self.display_table`. If a number is passed, the evaluation results will be truncated to that number before displayed.
            callback_metadata (dict): Metadata to be used for evaluate callback handlers.

        Returns:
            The evaluation results are returned as a dspy.EvaluationResult object containing the following attributes:

            - score: A float percentage score (e.g., 67.30) representing overall performance

            - results: a list of (example, prediction, score) tuples for each example in devset
        """
        metric, devset, num_threads, display_progress, display_table, save_as_csv, save_as_json = self._resolve_runtime_args(
            metric=metric,
            devset=devset,
            num_threads=num_threads,
            display_progress=display_progress,
            display_table=display_table,
            save_as_csv=save_as_csv,
            save_as_json=save_as_json,
            callback_metadata=callback_metadata,
        )

        tqdm.tqdm._instances.clear()

        executor = ParallelExecutor(
            num_threads=num_threads,
            disable_progress_bar=not display_progress,
            max_errors=(self.max_errors if self.max_errors is not None else dspy.settings.max_errors),
            provide_traceback=self.provide_traceback,
            compare_results=True,
        )

        def process_item(example):
            prediction = program(**example.inputs())
            score = metric(example, prediction)
            return prediction, score

        raw_results = executor.execute(process_item, devset)
        return self._finalize_results(
            raw_results=raw_results,
            devset=devset,
            metric=metric,
            display_table=display_table,
            save_as_csv=save_as_csv,
            save_as_json=save_as_json,
        )

    @with_callbacks
    async def acall(
        self,
        program: "dspy.Module",
        metric: Callable | None = None,
        devset: list["dspy.Example"] | None = None,
        num_threads: int | None = None,
        display_progress: bool | None = None,
        display_table: bool | int | None = None,
        callback_metadata: dict[str, Any] | None = None,
        save_as_csv: str | None = None,
        save_as_json: str | None = None,
    ) -> EvaluationResult:
        """Async version of __call__. Runs evaluation with native asyncio concurrency.

        Programs that define ``aforward`` are awaited directly. Sync-only programs are
        automatically wrapped via ``dspy.asyncify`` to run in a thread pool.

        Args and return value match ``__call__``.
        """
        metric, devset, num_threads, display_progress, display_table, save_as_csv, save_as_json = self._resolve_runtime_args(
            metric=metric,
            devset=devset,
            num_threads=num_threads,
            display_progress=display_progress,
            display_table=display_table,
            save_as_csv=save_as_csv,
            save_as_json=save_as_json,
            callback_metadata=callback_metadata,
        )

        tqdm.tqdm._instances.clear()
        effective_num_threads = num_threads or dspy.settings.num_threads
        effective_max_errors = self.max_errors if self.max_errors is not None else dspy.settings.max_errors

        with dspy.context(async_max_workers=effective_num_threads):
            raw_results = await self._execute_async(
                program=program,
                metric=metric,
                data=devset,
                num_threads=effective_num_threads,
                disable_progress_bar=not display_progress,
                max_errors=effective_max_errors,
                provide_traceback=self.provide_traceback,
            )

        return self._finalize_results(
            raw_results=raw_results,
            devset=devset,
            metric=metric,
            display_table=display_table,
            save_as_csv=save_as_csv,
            save_as_json=save_as_json,
        )

    def _resolve_runtime_args(
        self,
        *,
        metric: Callable | None,
        devset: list["dspy.Example"] | None,
        num_threads: int | None,
        display_progress: bool | None,
        display_table: bool | int | None,
        save_as_csv: str | None,
        save_as_json: str | None,
        callback_metadata: dict[str, Any] | None,
    ) -> tuple[Callable, list["dspy.Example"], int | None, bool, bool | int, str | None, str | None]:
        metric = metric if metric is not None else self.metric
        devset = devset if devset is not None else self.devset
        num_threads = num_threads if num_threads is not None else self.num_threads
        display_progress = display_progress if display_progress is not None else self.display_progress
        display_table = display_table if display_table is not None else self.display_table
        save_as_csv = save_as_csv if save_as_csv is not None else self.save_as_csv
        save_as_json = save_as_json if save_as_json is not None else self.save_as_json

        if callback_metadata:
            logger.debug(f"Evaluate is called with callback metadata: {callback_metadata}")

        return metric, devset, num_threads, display_progress, display_table, save_as_csv, save_as_json

    @staticmethod
    def _metric_name(metric: Callable) -> str:
        return metric.__name__ if isinstance(metric, types.FunctionType) else metric.__class__.__name__

    def _finalize_results(
        self,
        *,
        raw_results: list[Any],
        devset: list["dspy.Example"],
        metric: Callable,
        display_table: bool | int,
        save_as_csv: str | None,
        save_as_json: str | None,
    ) -> EvaluationResult:
        assert len(devset) == len(raw_results)
        raw_results = [((dspy.Prediction(), self.failure_score) if r is None else r) for r in raw_results]
        results = [(example, prediction, score) for example, (prediction, score) in zip(devset, raw_results, strict=False)]
        ncorrect, ntotal = sum(score for *_, score in results), len(devset)

        logger.info(f"Average Metric: {ncorrect} / {ntotal} ({round(100 * ncorrect / ntotal, 1)}%)")

        if display_table:
            if importlib.util.find_spec("pandas") is not None:
                metric_name = self._metric_name(metric)
                result_df = self._construct_result_table(results, metric_name)
                self._display_result_table(result_df, display_table, metric_name)
            else:
                logger.warning("Skipping table display since `pandas` is not installed.")

        if save_as_csv:
            metric_name = self._metric_name(metric)
            data = self._prepare_results_output(results, metric_name)
            with open(save_as_csv, "w", newline="") as csvfile:
                fieldnames = data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in data:
                    writer.writerow(row)
        if save_as_json:
            metric_name = self._metric_name(metric)
            data = self._prepare_results_output(results, metric_name)
            with open(save_as_json, "w") as f:
                json.dump(data, f)

        return EvaluationResult(score=round(100 * ncorrect / ntotal, 2), results=results)

    @staticmethod
    def _is_async_callable(fn: Any) -> bool:
        return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__call__", None))

    async def _invoke_program_async(self, program: Any, inputs: dict[str, Any]) -> Any:
        if hasattr(program, "aforward"):
            return await program.acall(**inputs)

        if self._is_async_callable(program):
            return await program(**inputs)

        return await dspy.asyncify(program)(**inputs)

    async def _invoke_metric_async(self, metric: Callable, example: Any, prediction: Any) -> Any:
        if self._is_async_callable(metric):
            return await metric(example, prediction)

        return metric(example, prediction)

    @staticmethod
    def _update_progress_bar(pbar, results):
        vals = [r[-1] for r in results if r is not None]
        nresults = sum(vals)
        metric_denominator = len(vals)
        pct = round(100 * nresults / metric_denominator, 1) if metric_denominator else 0
        pbar.set_description(f"Average Metric: {nresults:.2f} / {metric_denominator} ({pct}%)")
        pbar.update()

    async def _execute_async(
        self,
        *,
        program: Any,
        metric: Callable,
        data: list[Any],
        num_threads: int,
        disable_progress_bar: bool,
        max_errors: int,
        provide_traceback: bool | None,
    ) -> list[Any]:
        results: list[Any] = [None] * len(data)
        error_count = 0
        cancel_jobs = asyncio.Event()
        semaphore = asyncio.Semaphore(num_threads)

        pbar = tqdm.tqdm(
            total=len(data),
            dynamic_ncols=True,
            disable=disable_progress_bar,
            file=sys.stdout,
        )

        async def process_item(index: int, example: Any):
            nonlocal error_count

            if cancel_jobs.is_set():
                return index, None

            async with semaphore:
                if cancel_jobs.is_set():
                    return index, None

                try:
                    prediction = await self._invoke_program_async(program, example.inputs())
                    score = await self._invoke_metric_async(metric, example, prediction)
                    return index, (prediction, score)
                except Exception as e:
                    error_count += 1
                    if error_count >= max_errors:
                        cancel_jobs.set()
                    if provide_traceback:
                        logger.error(f"Error for {example}: {e}\n{traceback.format_exc()}")
                    else:
                        logger.error(f"Error for {example}: {e}. Set `provide_traceback=True` for traceback.")
                    return index, None

        tasks = [asyncio.create_task(process_item(index, item)) for index, item in enumerate(data)]
        try:
            for completed_task in asyncio.as_completed(tasks):
                try:
                    index, outcome = await completed_task
                except asyncio.CancelledError:
                    continue

                if outcome is not None and results[index] is None:
                    results[index] = outcome

                self._update_progress_bar(pbar, results)

                if cancel_jobs.is_set():
                    for task in tasks:
                        if not task.done():
                            task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            pbar.close()

        if cancel_jobs.is_set():
            logger.warning("Execution cancelled due to errors or interruption.")
            raise RuntimeError("Execution cancelled due to errors or interruption.")

        return results

    @staticmethod
    def _prepare_results_output(
            results: list[tuple["dspy.Example", "dspy.Example", Any]], metric_name: str
    ):
        return [
            (
                merge_dicts(example, prediction) | {metric_name: score}
                if prediction_is_dictlike(prediction)
                else example.toDict() | {"prediction": prediction, metric_name: score}
            )
            for example, prediction, score in results
        ]

    def _construct_result_table(
        self, results: list[tuple["dspy.Example", "dspy.Example", Any]], metric_name: str
    ) -> "pd.DataFrame":
        """
        Construct a pandas DataFrame from the specified result list.
        Let's not try to change the name of this method as it may be patched by external tracing tools.

        Args:
            results: The list of results to construct the result DataFrame from.
            metric_name: The name of the metric used for evaluation.

        Returns:
            The constructed pandas DataFrame.
        """
        import pandas as pd

        data = self._prepare_results_output(results, metric_name)

        # Truncate every cell in the DataFrame (DataFrame.applymap was renamed to DataFrame.map in Pandas 2.1.0)
        result_df = pd.DataFrame(data)
        result_df = result_df.map(truncate_cell) if hasattr(result_df, "map") else result_df.applymap(truncate_cell)

        return result_df.rename(columns={"correct": metric_name})

    def _display_result_table(self, result_df: "pd.DataFrame", display_table: bool | int, metric_name: str):
        """
        Display the specified result DataFrame in a table format.

        Args:
            result_df: The result DataFrame to display.
            display_table: Whether to display the evaluation results in a table.
                If a number is passed, the evaluation results will be truncated to that number before displayed.
            metric_name: The name of the metric used for evaluation.
        """
        if isinstance(display_table, bool):
            df_to_display = result_df.copy()
            truncated_rows = 0
        else:
            df_to_display = result_df.head(display_table).copy()
            truncated_rows = len(result_df) - display_table

        df_to_display = stylize_metric_name(df_to_display, metric_name)

        display_dataframe(df_to_display)

        if truncated_rows > 0:
            # Simplified message about the truncated rows
            message = f"""
            <div style='
                text-align: center;
                font-size: 16px;
                font-weight: bold;
                color: #555;
                margin: 10px 0;'>
                ... {truncated_rows} more rows not displayed ...
            </div>
            """
            display(HTML(message))


def prediction_is_dictlike(prediction):
    # Downstream logic for displaying dictionary-like predictions depends solely on the predictions
    # having a method called `items()` for iterating through key/value pairs
    return hasattr(prediction, "items") and callable(prediction.items)


def merge_dicts(d1, d2) -> dict:
    # Convert to dict if objects have toDict method (e.g., Example objects)
    if hasattr(d1, "toDict"):
        d1 = d1.toDict()
    if hasattr(d2, "toDict"):
        d2 = d2.toDict()

    merged = {}
    for k, v in d1.items():
        if k in d2:
            merged[f"example_{k}"] = v
        else:
            merged[k] = v

    for k, v in d2.items():
        if k in d1:
            merged[f"pred_{k}"] = v
        else:
            merged[k] = v

    return merged


def truncate_cell(content) -> str:
    """Truncate content of a cell to 25 words."""
    words = str(content).split()
    if len(words) > 25:
        return " ".join(words[:25]) + "..."
    return content


def stylize_metric_name(df: "pd.DataFrame", metric_name: str) -> "pd.DataFrame":
    """
    Stylize the cell contents of a pandas DataFrame corresponding to the specified metric name.

    :param df: The pandas DataFrame for which to stylize cell contents.
    :param metric_name: The name of the metric for which to stylize DataFrame cell contents.
    """
    def format_metric(x):
        if isinstance(x, float):
            return f"✔️ [{x:.3f}]"
        elif x is not None:
            return f"✔️ [{x}]"
        else:
            return ""
    df[metric_name] = df[metric_name].apply(format_metric)
    return df


def display_dataframe(df: "pd.DataFrame"):
    """
    Display the specified Pandas DataFrame in the console.

    :param df: The Pandas DataFrame to display.
    """
    import pandas as pd

    if is_in_ipython_notebook_environment():
        display(configure_dataframe_for_ipython_notebook_display(df))
    else:
        # Pretty print the DataFrame to the console
        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None
        ):  # more options can be specified also
            print(df)


def configure_dataframe_for_ipython_notebook_display(df: "pd.DataFrame") -> "pd.DataFrame":
    """Set various pandas display options for DataFrame in an IPython notebook environment."""
    import pandas as pd

    pd.options.display.max_colwidth = 70
    return df


def is_in_ipython_notebook_environment():
    """
    Check if the current environment is an IPython notebook environment.

    :return: True if the current environment is an IPython notebook environment, False otherwise.
    """
    try:
        from IPython import get_ipython

        # This is a best-effort check to see if we are in an IPython notebook environment
        return "IPKernelApp" in getattr(get_ipython(), "config", {})
    except ImportError:
        return False


# FIXME: TODO: The merge_dicts stuff above is way too quick and dirty.
# TODO: the display_table can't handle False but can handle 0!
# Not sure how it works with True exactly, probably fails too.
