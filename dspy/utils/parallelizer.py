import sys
import tqdm
import dspy
import signal
import threading
import traceback
import contextlib

from tqdm.contrib.logging import logging_redirect_tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class ParallelExecutor:
    def __init__(
        self,
        num_threads,
        max_errors=5,
        display_progress=False,
        provide_traceback=False,
        compare_results=False,
    ):
        self.num_threads = num_threads
        self.display_progress = display_progress
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
        def wrapped(item):
            thread_stacks = dspy.settings.stack_by_thread
            creating_new_thread = threading.get_ident() not in thread_stacks
            if creating_new_thread:
                thread_stacks[threading.get_ident()] = list(dspy.settings.main_stack)

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
                    dspy.logger.error(
                        f"Error processing item {item}: {e}\nStack trace:\n{traceback.format_exc()}"
                    )
                else:
                    dspy.logger.error(
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
            disable=not self.display_progress,
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
            dspy.logger.warning("Execution was cancelled due to errors.")
            raise Exception("Execution was cancelled due to errors.")
        return results


    def _update_progress(self, pbar, nresults, ntotal):
        if self.compare_results:
            pbar.set_description(f"Average Metric: {nresults:.2f} / {ntotal} ({round(100 * nresults / ntotal, 1)}%)")
        else:
            pbar.set_description(f"Processed {nresults} / {ntotal} examples")
        pbar.update()


    def _execute_multi_thread(self, function, data):
        results = [None] * len(data)  # Pre-allocate results list to maintain order
        job_cancelled = "cancelled"

        @contextlib.contextmanager
        def interrupt_handler_manager():
            """Sets the cancel_jobs event when a SIGINT is received."""
            default_handler = signal.getsignal(signal.SIGINT)

            def interrupt_handler(sig, frame):
                self.cancel_jobs.set()
                dspy.logger.warning("Received SIGINT. Cancelling execution.")
                default_handler(sig, frame)

            signal.signal(signal.SIGINT, interrupt_handler)
            yield
            # reset to the default handler
            signal.signal(signal.SIGINT, default_handler)

        def cancellable_function(index_item):
            index, item = index_item
            if self.cancel_jobs.is_set():
                return index, job_cancelled
            return index, function(item)

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor, interrupt_handler_manager():
            futures = {executor.submit(cancellable_function, pair): pair for pair in enumerate(data)}
            pbar = tqdm.tqdm(
                total=len(data),
                dynamic_ncols=True,
                disable=not self.display_progress,
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
            dspy.logger.warning("Execution was cancelled due to errors.")
            raise Exception("Execution was cancelled due to errors.")
        return results
