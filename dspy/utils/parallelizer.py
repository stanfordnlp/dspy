import logging
import sys
import tqdm
import dspy
import signal
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
            return self._execute_single_thread(wrapped_function, data)
        else:
            return self._execute_multi_thread(wrapped_function, data)


    def _wrap_function(self, function):
        # Wrap the function with threading context and error handling
        def wrapped(item, parent_id=None):
            thread_stacks = dspy.settings.stack_by_thread
            current_thread_id = threading.get_ident()
            creating_new_thread = current_thread_id not in thread_stacks

            assert creating_new_thread or threading.get_ident() == dspy.settings.main_tid

            if creating_new_thread:
                # If we have a parent thread ID, copy its stack. TODO: Should the caller just pass a copy of the stack?
                if parent_id and parent_id in thread_stacks:
                    thread_stacks[current_thread_id] = list(thread_stacks[parent_id])
                else:
                    thread_stacks[current_thread_id] = list(dspy.settings.main_stack)

                # TODO: Consider the behavior below.
                # import copy; thread_stacks[current_thread_id].append(copy.deepcopy(thread_stacks[current_thread_id][-1]))

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
            finally:
                if creating_new_thread:
                    del thread_stacks[threading.get_ident()]
        return wrapped


    def _execute_single_thread(self, function, data):
        results = []
        pbar = tqdm.tqdm(
            total=len(data),
            dynamic_ncols=True,
            disable=self.disable_progress_bar,
            file=sys.stdout,
        )
        for item in data:
            with logging_redirect_tqdm():
                if self.cancel_jobs.is_set():
                    break
                result = function(item)
                results.append(result)
                if self.compare_results:
                    # Assumes score is the last element of the result tuple
                    self._update_progress(pbar, sum([r[-1] for r in results if r is not None]), len([r for r in data if r is not None]))
                else:
                    self._update_progress(pbar, len(results), len(data))
        pbar.close()
        if self.cancel_jobs.is_set():
            logger.warning("Execution was cancelled due to errors.")
            raise Exception("Execution was cancelled due to errors.")
        return results


    def _update_progress(self, pbar, nresults, ntotal):
        if self.compare_results:
            pbar.set_description(f"Average Metric: {nresults:.2f} / {ntotal} ({round(100 * nresults / ntotal, 1) if ntotal > 0 else 0}%)")
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
                    default_handler(sig, frame)

                signal.signal(signal.SIGINT, interrupt_handler)
                try:
                    yield
                finally:
                    signal.signal(signal.SIGINT, default_handler)
            else:
                # If not in the main thread, skip setting signal handlers
                yield

        def cancellable_function(index_item, parent_id=None):
            index, item = index_item
            if self.cancel_jobs.is_set():
                return index, job_cancelled
            return index, function(item, parent_id)
        
        parent_id = threading.get_ident() if threading.current_thread() is not threading.main_thread() else None

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor, interrupt_handler_manager():
            futures = {executor.submit(cancellable_function, pair, parent_id): pair for pair in enumerate(data)}
            pbar = tqdm.tqdm(
                total=len(data),
                dynamic_ncols=True,
                disable=self.disable_progress_bar,
                file=sys.stdout,
            )

            for future in as_completed(futures):
                index, result = future.result()
            
                if result is job_cancelled:
                    continue
                results[index] = result

                if self.compare_results:
                    # Assumes score is the last element of the result tuple
                    self._update_progress(pbar, sum([r[-1] for r in results if r is not None]), len([r for r in results if r is not None]))
                else:
                    self._update_progress(pbar, len([r for r in results if r is not None]), len(data))
            pbar.close()
        if self.cancel_jobs.is_set():
            logger.warning("Execution was cancelled due to errors.")
            raise Exception("Execution was cancelled due to errors.")
        return results
