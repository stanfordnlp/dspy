import threading
from typing import Any, List, Optional, Tuple

from dspy.dsp.utils.settings import settings
from dspy.primitives.example import Example
from dspy.utils.parallelizer import ParallelExecutor


class Parallel:
    def __init__(
        self,
        num_threads: Optional[int] = None,
        max_errors: int = 10,
        access_examples: bool = True,
        return_failed_examples: bool = False,
        provide_traceback: Optional[bool] = None,
        disable_progress_bar: bool = False,
    ):
        super().__init__()
        self.num_threads = num_threads or settings.num_threads
        self.max_errors = max_errors
        self.access_examples = access_examples
        self.return_failed_examples = return_failed_examples
        self.provide_traceback = provide_traceback
        self.disable_progress_bar = disable_progress_bar

        self.error_count = 0
        self.error_lock = threading.Lock()
        self.cancel_jobs = threading.Event()
        self.failed_examples = []
        self.exceptions = []

    def forward(self, exec_pairs: List[Tuple[Any, Example]], num_threads: Optional[int] = None) -> List[Any]:
        num_threads = num_threads if num_threads is not None else self.num_threads

        executor = ParallelExecutor(
            num_threads=num_threads,
            max_errors=self.max_errors,
            provide_traceback=self.provide_traceback,
            disable_progress_bar=self.disable_progress_bar,
        )

        def process_pair(pair):
            result = None
            module, example = pair

            if isinstance(example, Example):
                if self.access_examples:
                    result = module(**example.inputs())
                else:
                    result = module(example)
            elif isinstance(example, dict):
                result = module(**example)
            elif isinstance(example, list) and module.__class__.__name__ == "Parallel":
                result = module(example)
            elif isinstance(example, tuple):
                result = module(*example)
            else:
                raise ValueError(
                    f"Invalid example type: {type(example)}, only supported types are Example, dict, list and tuple"
                )
            return result

        # Execute the processing function over the execution pairs
        results = executor.execute(process_pair, exec_pairs)

        if self.return_failed_examples:
            return results, self.failed_examples, self.exceptions
        else:
            return results

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
