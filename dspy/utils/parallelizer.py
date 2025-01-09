import sys
import tqdm
import signal
import logging
import threading
import traceback
import contextlib
from tqdm.contrib.logging import logging_redirect_tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        """Offers isolation between the tasks (dspy.settings) irrespective of whether num_threads == 1 or > 1."""
        self.num_threads = num_threads
        self.disable_progress_bar = disable_progress_bar
        self.max_errors = max_errors
        self.provide_traceback = provide_traceback
        self.compare_results = compare_results

        self.error_count = 0
        self.error_lock = threading.Lock()
        self.cancel_jobs = threading.Event()

    def execute(self, function, data):
        wrapped_function = self._wrap_function(function)
        if self.num_threads == 1:
            return self._execute_isolated_single_thread(wrapped_function, data)
        else:
            return self._execute_multi_thread(wrapped_function, data)

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
                with self.error_lock:
                    self.error_count += 1
                    if self.error_count >= self.max_errors:
                        raise e
                return None
            finally:
                thread_local_overrides.overrides = original_overrides

        return wrapped

    def _execute_isolated_single_thread(self, function, data):
        results = []
        pbar = tqdm.tqdm(total=len(data), dynamic_ncols=True, disable=self.disable_progress_bar, file=sys.stdout)

        try:
            with logging_redirect_tqdm():
                for item in data:
                    if self.cancel_jobs.is_set():
                        break

                    result = function(item)
                    results.append(result)

                    if self.compare_results:
                        rs = [r for r in results if r is not None]
                        # Assumes score is the last element of the result tuple
                        scores = [r[-1] for r in rs]
                        self._update_progress(pbar, sum(scores), len(rs))
                    else:
                        self._update_progress(pbar, len(results), len(data))

        finally:
            pbar.close()

        return results

    def _update_progress(self, pbar, nresults, ntotal):
        if self.compare_results:
            percentage = round(100 * nresults / ntotal, 1) if ntotal > 0 else 0
            pbar.set_description(f"Average Metric: {nresults:.2f} / {ntotal} ({percentage}%)")
        else:
            pbar.set_description(f"Processed {nresults} / {ntotal} examples")

        pbar.update()

    def _execute_multi_thread(self, function, data):
        @contextlib.contextmanager
        def interrupt_handler_manager():
            """Sets the cancel_jobs event when a SIGINT is received, only in the main thread."""

            # TODO: Is this check conducive to nested usage of ParallelExecutor?
            if threading.current_thread() is threading.main_thread():
                default_handler = signal.getsignal(signal.SIGINT)

                def interrupt_handler(sig, frame):
                    self.cancel_jobs.set()
                    logger.warning("Received SIGINT. Cancelling execution.")
                    # Re-raise the signal to allow default behavior
                    default_handler(sig, frame)

                signal.signal(signal.SIGINT, interrupt_handler)
                try:
                    yield
                finally:
                    signal.signal(signal.SIGINT, default_handler)
            else:
                # If not in the main thread, skip setting signal handless
                yield

        def thread_local_function(index_item):
            index, item = index_item
            return index, function(item)

        pbar = tqdm.tqdm(total=len(data), dynamic_ncols=True, disable=self.disable_progress_bar, file=sys.stdout)
        results = [None] * len(data)  # Pre-allocate results list to maintain order

        try:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor, interrupt_handler_manager():
                futures = {}
                for pair in enumerate(data):
                    future = executor.submit(thread_local_function, pair)
                    futures[future] = pair

                for future in as_completed(futures):
                    # from CTRL+C
                    if self.cancel_jobs.is_set():
                        break

                    index, result = future.result()
                    results[index] = result

                    rs = [r for r in results if r is not None]
                    if self.compare_results:
                        # Assumes score is the last element of the result tuple
                        scores = [r[-1] for r in rs]
                        self._update_progress(pbar, sum(scores), len(rs))
                    else:
                        self._update_progress(pbar, len(rs), len(data))
        finally:
            pbar.close()

        return results
