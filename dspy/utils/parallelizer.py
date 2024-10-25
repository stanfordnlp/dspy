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
    ):
        self.num_threads = num_threads
        self.display_progress = display_progress
        self.max_errors = max_errors
        self.provide_traceback = provide_traceback

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
                pbar.update()
        pbar.close()
        if self.cancel_jobs.is_set():
            dspy.logger.warning("Execution was cancelled due to errors.")
            raise Exception("Execution was cancelled due to errors.")
        return results


    def _execute_multi_thread(self, function, data):
        results = []
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

        def cancellable_function(item):
            if self.cancel_jobs.is_set():
                return job_cancelled
            return function(item)

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor, interrupt_handler_manager():
            futures = {executor.submit(cancellable_function, item): item for item in data}
            pbar = tqdm.tqdm(
                total=len(data),
                dynamic_ncols=True,
                disable=not self.display_progress,
                file=sys.stdout,
            )
            for future in as_completed(futures):
                result = future.result()
                if result is job_cancelled:
                    continue
                results.append(result)
                pbar.update()
            pbar.close()
        if self.cancel_jobs.is_set():
            dspy.logger.warning("Execution was cancelled due to errors.")
            raise Exception("Execution was cancelled due to errors.")
        return results
