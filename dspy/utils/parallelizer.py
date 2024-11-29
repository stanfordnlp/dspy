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
        # Wrap the function with error handling
        def wrapped(item):
            if self.cancel_jobs.is_set():
                return None
            try:
                return function(item)
            except Exception as e:
                with self.error_lock:
                    self.error_count += 1
                    current_error_count = self.error_count
                if current_error_count >= self.max_errors:
                    self.cancel_jobs.set()
                    raise e
                if self.provide_traceback:
                    logger.error(
                        f"Error processing item {item}: {e}\nStack trace:\n{traceback.format_exc()}"
                    )
                else:
                    logger.error(
                        f"Error processing item {item}: {e}. Set `provide_traceback=True` to see the stack trace."
                    )
                return None
        return wrapped

    def _execute_isolated_single_thread(self, function, data):
        results = []
        pbar = tqdm.tqdm(
            total=len(data),
            dynamic_ncols=True,
            disable=self.disable_progress_bar,
            file=sys.stdout
        )

        for item in data:
            with logging_redirect_tqdm():
                if self.cancel_jobs.is_set():
                    break

                # Create an isolated context for each task using thread-local overrides
                from dsp.utils.settings import thread_local_overrides
                original_overrides = thread_local_overrides.overrides
                thread_local_overrides.overrides = thread_local_overrides.overrides.copy()

                try:
                    result = function(item)
                    results.append(result)
                finally:
                    thread_local_overrides.overrides = original_overrides

                if self.compare_results:
                    # Assumes score is the last element of the result tuple
                    self._update_progress(
                        pbar,
                        sum([r[-1] for r in results if r is not None]),
                        len([r for r in data if r is not None]),
                    )
                else:
                    self._update_progress(pbar, len(results), len(data))

        pbar.close()

        if self.cancel_jobs.is_set():
            logger.warning("Execution was cancelled due to errors.")
            raise Exception("Execution was cancelled due to errors.")

        return results

    def _update_progress(self, pbar, nresults, ntotal):
        if self.compare_results:
            percentage = round(100 * nresults / ntotal, 1) if ntotal > 0 else 0
            pbar.set_description(f"Average Metric: {nresults:.2f} / {ntotal} ({percentage}%)")
        else:
            pbar.set_description(f"Processed {nresults} / {ntotal} examples")

        pbar.update()

    def _execute_multi_thread(self, function, data):
        results = [None] * len(data)  # Pre-allocate results list to maintain order
        job_cancelled = "cancelled"

        @contextlib.contextmanager
        def interrupt_handler_manager():
            """Sets the cancel_jobs event when a SIGINT is received, only in the main thread."""
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
                # If not in the main thread, skip setting signal handlers
                yield

        def cancellable_function(parent_overrides, index_item):
            index, item = index_item
            if self.cancel_jobs.is_set():
                return index, job_cancelled

            # Create an isolated context for each task using thread-local overrides
            from dsp.utils.settings import thread_local_overrides
            original_overrides = thread_local_overrides.overrides
            thread_local_overrides.overrides = parent_overrides.copy()

            try:
                return index, function(item)
            finally:
                thread_local_overrides.overrides = original_overrides

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor, interrupt_handler_manager():
            # Capture the parent thread's overrides
            from dsp.utils.settings import thread_local_overrides
            parent_overrides = thread_local_overrides.overrides.copy()

            futures = {}
            for pair in enumerate(data):
                # Pass the parent thread's overrides to each thread
                future = executor.submit(cancellable_function, parent_overrides, pair)
                futures[future] = pair

            pbar = tqdm.tqdm(
                total=len(data),
                dynamic_ncols=True,
                disable=self.disable_progress_bar,
                file=sys.stdout
            )

            for future in as_completed(futures):
                index, result = future.result()

                if result is job_cancelled:
                    continue

                results[index] = result

                if self.compare_results:
                    # Assumes score is the last element of the result tuple
                    self._update_progress(
                        pbar,
                        sum([r[-1] for r in results if r is not None]),
                        len([r for r in results if r is not None]),
                    )
                else:
                    self._update_progress(
                        pbar,
                        len([r for r in results if r is not None]),
                        len(data),
                    )

            pbar.close()

        if self.cancel_jobs.is_set():
            logger.warning("Execution was cancelled due to errors.")
            raise Exception("Execution was cancelled due to errors.")

        return results
