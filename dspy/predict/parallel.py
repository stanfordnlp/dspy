import dspy
import threading
import traceback

from typing import Tuple, List, Any

from dspy.utils.parallelizer import ParallelExecutor


class Parallel(dspy.Module):
    def __init__(
        self,
        num_threads: int = 32,
        max_errors: int = 10,
        return_failed_examples: bool = False,
        provide_traceback: bool = False,
    ):
        super().__init__()
        self.num_threads = num_threads
        self.max_errors = max_errors
        self.return_failed_examples = return_failed_examples
        self.provide_traceback = provide_traceback

        self.error_count = 0
        self.error_lock = threading.Lock()
        self.cancel_jobs = threading.Event()
        self.failed_examples = []
        self.exceptions = []


    def forward(self, exec_pairs: List[Tuple[dspy.Module, dspy.Example]], num_threads: int = None) -> List[Any]:
        num_threads = num_threads if num_threads is not None else self.num_threads

        executor = ParallelExecutor(
            num_threads=num_threads,
            display_progress=True,
            max_errors=self.max_errors,
            provide_traceback=self.provide_traceback,
        )

        def process_pair(pair):
            module, example = pair
            thread_stacks = dspy.settings.stack_by_thread
            creating_new_thread = threading.get_ident() not in thread_stacks
            if creating_new_thread:
                thread_stacks[threading.get_ident()] = list(dspy.settings.main_stack)

            try:
                result = module(**example.inputs())
                return result
            except Exception as e:
                with self.error_lock:
                    self.error_count += 1
                    current_error_count = self.error_count

                if self.return_failed_examples:
                    self.failed_examples.append(pair)
                    self.exceptions.append(e)

                if current_error_count >= self.max_errors:
                    self.cancel_jobs.set()
                    raise e

                if self.provide_traceback:
                    dspy.logger.error(
                        f"Error processing pair {pair}: {e}\nStack trace:\n{traceback.format_exc()}"
                    )
                else:
                    dspy.logger.error(
                        f"Error processing pair {pair}: {e}. Set `provide_traceback=True` to see the stack trace."
                    )
                return None
            finally:
                if creating_new_thread:
                    del thread_stacks[threading.get_ident()]

        # Execute the processing function over the execution pairs
        results = executor.execute(process_pair, exec_pairs)

        if self.return_failed_examples:
            return results, self.failed_examples, self.exceptions
        else:
            return results
