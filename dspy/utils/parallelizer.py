from concurrent.futures import ThreadPoolExecutor
import logging
import multiprocessing as mp
import sys
import threading
import time
import traceback

from tqdm.contrib.logging import logging_redirect_tqdm
import tqdm

logger = logging.getLogger(__name__)


class ParallelExecutor:
    def __init__(
        self,
        num_threads,
        max_errors=5,
        disable_progress_bar=False,
        provide_traceback=False,
        compare_results=False,
    ):
        assert num_threads > 0

        """Offers isolation between the tasks (dspy.settings) irrespective of whether num_threads == 1 or > 1."""
        self.num_threads = num_threads
        self.disable_progress_bar = disable_progress_bar
        self.max_errors = max_errors
        self.provide_traceback = provide_traceback
        self.compare_results = compare_results

        self.error_count = 0
        self._lock = threading.Lock()

    def execute(self, function, data):
        wrapped_function = self._wrap_function(function)
        exec_type = "multi" if self.num_threads != 1 else "single"
        executor = getattr(self, f"_execute_{exec_type}_thread")
        return executor(wrapped_function, data)

    def _wrap_function(self, function):
        from dspy.dsp.utils.settings import thread_local_overrides

        # Wrap the function with error handling
        def wrapped(item):
            original_overrides = thread_local_overrides.overrides
            thread_local_overrides.overrides = thread_local_overrides.overrides.copy()
            try:
                return function(item)
            except Exception as e:
                if self.provide_traceback:
                    logger.error(f"Error processing item {item}: {e}\nStack trace:\n{traceback.format_exc()}")
                else:
                    logger.error(
                        f"Error processing item {item}: {e}. Set `provide_traceback=True` to see the stack trace."
                    )
                with self._lock:
                    self.error_count += 1
                    if self.error_count >= self.max_errors:
                        raise e
                return None
            finally:
                thread_local_overrides.overrides = original_overrides

        return wrapped

    def _create_pbar(self, data: list):
        return tqdm.tqdm(total=len(data), dynamic_ncols=True, disable=self.disable_progress_bar, file=sys.stdout)

    def _update_pbar(self, pbar: tqdm.tqdm, nresults, ntotal):
        if self.compare_results:
            percentage = round(100 * nresults / ntotal, 1) if ntotal > 0 else 0
            pbar.set_description(f"Average Metric: {nresults:.2f} / {ntotal} ({percentage}%)", refresh=True)
        else:
            pbar.set_description(f"Processed {nresults} / {ntotal} examples", refresh=True)

    def _execute_single_thread(self, function, data):
        total_score = 0
        total_processed = 0

        def function_with_progress(item):
            result = function(item)

            nonlocal total_score, total_processed, pbar
            total_processed += 1
            if self.compare_results:
                if result is not None:
                    total_score += result[-1]
                self._update_pbar(pbar, total_score, total_processed)
            else:
                self._update_pbar(pbar, total_processed, len(data))

            return result

        with self._create_pbar(data) as pbar, logging_redirect_tqdm():
            return list(map(function_with_progress, data))

    def _execute_multi_thread(self, function, data):
        pbar = self._create_pbar(data)
        total_score = 0
        total_processed = 0

        def function_with_progress(item):
            result = function(item)

            nonlocal total_score, total_processed, pbar
            total_processed += 1
            if self.compare_results:
                if result is not None:
                    total_score += result[-1]
                self._update_pbar(pbar, total_score, total_processed)
            else:
                self._update_pbar(pbar, total_processed, len(data))

            return result

        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            try:
                return list(pool.map(function_with_progress, data))
            except Exception:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            finally:
                pbar.close()
