import sys
import threading
import types
from collections.abc import Callable
from typing import Literal, List, Tuple, Optional, Iterable, overload, Union, Any

# noinspection PyUnresolvedReferences
import pandas as pd
# noinspection PyUnresolvedReferences
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import dspy
from dspy import Example, Prediction

try:
    # noinspection PyPackageRequirements
    from IPython.display import HTML  # type: ignore
    # noinspection PyPackageRequirements
    from IPython.display import display as ipython_display  # type: ignore
except ImportError:
    ipython_display = print
    def HTML(x): # noqa - linters dislike upper case function name
        return x

from concurrent.futures import ThreadPoolExecutor, as_completed

from dsp.evaluation.utils import *

"""
TODO: Counting failures and having a max_failure count. When that is exceeded (also just at the end),
we print the number of failures, the first N examples that failed, and the first N exceptions raised.
"""


class Evaluate:
    """
    Reification of the process of evaluating a model.  Invoked using its
    `__call__` method.

    Parameters
    ==========
    devset: Iterable[Example]
       An iterable of Examples
    metric: Callable, optional
       A function that takes an Example and a Prediction and keyword args as arguments
       and returns a boolean
    num_threads: int, optional
       How many
    display_progress: bool, optional
    display_table: Union[bool, int], optional
        This is either a boolean indicating whether or not to display a table of
        results or an integer indicating how many rows of results to display.
    max_errors: int, optional
    return_outputs: bool, optional
        Whether or not to return the predictions.
    """

    def __init__(
            self,
            *,
            devset: Iterable[Example],
            metric: Optional[Callable] = None,
            num_threads=1,
            display_progress=False,
            display_table: Union[bool, int] = False,
            max_errors=5,
            return_outputs=False,
            **_kwargs,
    ):
        self.devset = devset
        self.metric = metric
        self.num_threads = num_threads
        self.display_progress = display_progress
        self.display_table = display_table
        self.max_errors = max_errors
        self.error_count = 0
        self.error_lock = threading.Lock()
        self.return_outputs = return_outputs

        if "display" in _kwargs:
            dspy.logger.warning(
                ("DeprecationWarning: 'display' has been deprecated. "
                 "To see all information for debugging, use 'dspy.set_log_level('debug')'. "
                 "In the future this will raise an error."
                 )
            )

    def _execute_single_thread(self, wrapped_program, devset, display_progress):
        ncorrect = 0
        ntotal = 0
        reordered_devset = []

        pbar = tqdm.tqdm(
            total=len(devset),
            dynamic_ncols=True,
            disable=not display_progress,
            file=sys.stdout,
        )
        for idx, arg in devset:
            with logging_redirect_tqdm():
                example_idx, example, prediction, score = wrapped_program(idx, arg)
                reordered_devset.append((example_idx, example, prediction, score))
                ncorrect += score
                ntotal += 1
                self._update_progress(pbar, ncorrect, ntotal)

        pbar.close()

        return reordered_devset, ncorrect, ntotal

    def _execute_multi_thread(
            self, wrapped_program, devset, num_threads, display_progress
    ):
        ncorrect = 0
        ntotal = 0
        reordered_devset = []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(wrapped_program, idx, arg) for idx, arg in devset
            }
            pbar = tqdm.tqdm(
                total=len(devset), dynamic_ncols=True, disable=not display_progress
            )

            for future in as_completed(futures):
                example_idx, example, prediction, score = future.result()
                reordered_devset.append((example_idx, example, prediction, score))
                ncorrect += score
                ntotal += 1
                self._update_progress(pbar, ncorrect, ntotal)
            pbar.close()

        return reordered_devset, ncorrect, ntotal

    @staticmethod
    def _update_progress(pbar, ncorrect, ntotal):
        pbar.set_description(
            f"Average Metric: {ncorrect} / {ntotal}  ({round(100 * ncorrect / ntotal, 1)})"
        )
        pbar.update()

    ### Type stubs below for three alternative calling patterns, depending
    ### on the values of `return_all_scores` and `return_outputs`
    @overload
    def __call__(
            self,
            program,
            metric=...,
            devset: Optional[Iterable] = ...,  # Needs more specific type, if possible
            num_threads: Optional[int] = ...,
            display_progress: Optional[bool] = ...,
            display_table: Union[bool, int, None] = ...,
            *,
            return_all_scores: Literal[True],
            return_outputs: Literal[True],
    ) -> Tuple[float, List, List[float]]:
        ...

    @overload
    def __call__(
            self,
            program,
            metric=...,
            devset: Optional[Iterable] = ...,  # Needs more specific type, if possible
            num_threads: Optional[int] = ...,
            display_progress: Optional[bool] = ...,
            display_table: Union[bool, int, None] = ...,
            *,
            return_all_scores: Literal[True],
            return_outputs: Literal[False],
    ) -> Tuple[float, List[float]]:
        ...

    @overload
    def __call__(
            self,
            program,
            metric=...,
            devset: Optional[Iterable] = ...,  # Needs more specific type, if possible
            num_threads: Optional[int] = ...,
            display_progress: Optional[bool] = ...,
            display_table: Union[bool, int, None] = ...,
            *,
            return_all_scores: Literal[False],
            return_outputs: Literal[True],
    ) -> Tuple[float, List[Tuple[Example, Prediction, Any]]]:
        ...

    @overload
    def __call__(
            self,
            program,
            metric=...,
            devset: Optional[Iterable] = ...,  # Needs more specific type, if possible
            num_threads: Optional[int] = ...,
            display_progress: Optional[bool] = ...,
            display_table: Union[bool, int, None] = ...,
            *,
            return_all_scores: Literal[False],
            return_outputs: Literal[False],
    ) -> float:
        ...

    def __call__(
            self,
            program,
            metric: Optional[Callable] = None,
            devset: Optional[Iterable] = None,  # Needs more specific type, if possible
            num_threads: Optional[int] = None,
            display_progress: Optional[bool] = None,
            display_table: Union[bool, int, None] = None,
            return_all_scores: bool = False,
            return_outputs: bool = False,
    ):
        """
        Returns
        =======
        score: float, results: list of Prediction, all_scores: list of scores
            if `return_all_scores` and `return_outputs` are both True.
        score: float, all_scores: list of scores
            if `return_all_scores` is True and `return_outputs` is True.
        score: float, results: list of Prediction
            if `return_all_scores` is False and `return_outputs` is True.
        score: float
            if both flags are false
        """
        metric = metric if metric is not None else self.metric
        devset = devset if devset is not None else self.devset
        num_threads = num_threads if num_threads is not None else self.num_threads
        display_progress = (
            display_progress if display_progress is not None else self.display_progress
        )
        display_table = (
            display_table if display_table is not None else self.display_table
        )
        return_outputs = (
            return_outputs if return_outputs is not False else self.return_outputs
        )
        results = []

        def wrapped_program(example_idx, example):
            # NOTE: TODO: Won't work if threads create threads!
            thread_stacks = dspy.settings.stack_by_thread
            creating_new_thread = threading.get_ident() not in thread_stacks
            if creating_new_thread:
                thread_stacks[threading.get_ident()] = list(dspy.settings.main_stack)

            try:
                prediction = program(**example.inputs())
                score = metric(
                    example,
                    prediction,
                )  # FIXME: TODO: What's the right order? Maybe force name-based kwargs!

                # increment assert and suggest failures to program's attributes
                if hasattr(program, "_assert_failures"):
                    # noinspection PyProtectedMember
                    program._assert_failures += dspy.settings.get("assert_failures")
                if hasattr(program, "_suggest_failures"):
                    # noinspection PyProtectedMember
                    program._suggest_failures += dspy.settings.get("suggest_failures")

                return example_idx, example, prediction, score
            except Exception as e:
                with self.error_lock:
                    self.error_count += 1
                    current_error_count = self.error_count
                if current_error_count >= self.max_errors:
                    raise e

                dspy.logger.error(f"Error for example in dev set: \t\t {e}")

                return example_idx, example, {}, 0.0
            finally:
                if creating_new_thread:
                    del thread_stacks[threading.get_ident()]

        devset = list(enumerate(devset))  # This actually changes the type of `devset`
        # noinspection PyProtectedMember
        tqdm.tqdm._instances.clear()  # type: ignore

        if num_threads == 1:
            reordered_devset, ncorrect, ntotal = self._execute_single_thread(
                wrapped_program, devset, display_progress
            )
        else:
            reordered_devset, ncorrect, ntotal = self._execute_multi_thread(
                wrapped_program,
                devset,
                num_threads,
                display_progress,
            )

        dspy.logger.info(
            f"Average Metric: {ncorrect} / {ntotal} ({round(100 * ncorrect / ntotal, 1)}%)"
        )

        predicted_devset = sorted(reordered_devset)

        if return_outputs:  # Handle the return_outputs logic
            results = [
                (example, prediction, score)
                for _, example, prediction, score in predicted_devset
            ]

        data = [
            merge_dicts(example, prediction) | {"correct": score}
            for _, example, prediction, score in predicted_devset
        ]

        df = pd.DataFrame(data)

        # Truncate every cell in the DataFrame
        if hasattr(
                df, "map"
        ):  # DataFrame.applymap was renamed to DataFrame.map in Pandas 2.1.0
            df = df.map(truncate_cell)
        else:
            df = df.applymap(truncate_cell)  # type: ignore

        # Rename the 'correct' column to the name of the metric object
        metric_name = (
            metric.__name__
            if isinstance(metric, types.FunctionType)
            else metric.__class__.__name__
        )
        df.rename(columns={"correct": metric_name}, inplace=True)

        if display_table:
            if isinstance(display_table, int):
                df_to_display = df.head(display_table).copy()
                truncated_rows = len(df) - display_table
            else:
                df_to_display = df.copy()
                truncated_rows = 0

            styled_df = configure_dataframe_display(df_to_display, metric_name)

            ipython_display(styled_df)

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
                ipython_display(HTML(message))

        if return_all_scores and return_outputs:
            return (
                round(100 * ncorrect / ntotal, 2),
                results,
                [score for *_, score in predicted_devset],
            )
        elif return_all_scores:
            return round(100 * ncorrect / ntotal, 2), [
                score for *_, score in predicted_devset
            ]
        elif return_outputs:
            return round(100 * ncorrect / ntotal, 2), results

        return round(100 * ncorrect / ntotal, 2)


def merge_dicts(d1, d2):
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


def truncate_cell(content):
    """Truncate content of a cell to 25 words."""
    words = str(content).split()
    if len(words) > 25:
        return " ".join(words[:25]) + "..."
    return content


def configure_dataframe_display(df, metric_name):
    """Set various pandas display options for DataFrame."""
    pd.options.display.max_colwidth = None
    pd.set_option("display.max_colwidth", 20)  # Adjust the number as needed
    pd.set_option("display.width", 400)  # Adjust

    # df[metric_name] = df[metric_name].apply(lambda x: f'✔️ [{x}]' if x is True else f'❌ [{x}]')
    # df.loc[:, metric_name] = df[metric_name].apply(lambda x: f"✔️ [{x}]" if x is True else f"{x}")
    df[metric_name] = df[metric_name].apply(lambda x: f"✔️ [{x}]" if x else str(x))

    # Return styled DataFrame
    return df.style.set_table_styles(
        [
            {"selector": "th", "props": [("text-align", "left")]},
            {"selector": "td", "props": [("text-align", "left")]},
        ],
    ).set_properties(
        **{
            "text-align": "left",
            "white-space": "pre-wrap",
            "word-wrap": "break-word",
            "max-width": "400px",
        },
    )

# FIXME: TODO: The merge_dicts stuff above is way too quick and dirty.
# TODO: the display_table can't handle False but can handle 0! Not sure how it works
#       with True exactly, probably fails too.
