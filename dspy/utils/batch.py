"""Coalesce synchronous LM calls without imposing lockstep program execution."""

import math
import threading
from concurrent.futures import Future, TimeoutError

from dspy.utils.exceptions import LMProviderError


class BatchCoordinator:
    """One dispatcher per Parallel invocation; the existing executor owns workers.

    A short collection window batches ready calls, not a fixed number of examples.
    Workers may finish, fail, hit a cache, or make additional calls independently.
    Providers must honor cancellation and bound their network operations.
    """

    def __init__(self, cancel_event, timeout, poll_interval):
        if not all(math.isfinite(value) and value > 0 for value in (timeout, poll_interval)):
            raise ValueError("batch_timeout and batch_poll_interval must be finite and positive")
        self.cancel_event = cancel_event
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.condition = threading.Condition()
        self.pending = []
        self.closed = False
        self.thread = threading.Thread(target=self._dispatch, name="dspy-batch-dispatch")

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        with self.condition:
            self.closed = True
            self.cancel_event.set()
            self.condition.notify_all()
        self.thread.join()

    def submit(self, lm, request):
        future = Future()
        with self.condition:
            if self.closed or self.cancel_event.is_set():
                raise LMProviderError("Provider batch execution cancelled")
            self.pending.append((lm, request, future))
            self.condition.notify()
        while True:
            try:
                return future.result(timeout=0.1)
            except TimeoutError:
                if future.done():
                    # The provider's outcome itself may be a TimeoutError.
                    return future.result()
                if self.cancel_event.is_set():
                    raise LMProviderError("Provider batch execution cancelled")

    def _dispatch(self):
        calls = []
        try:
            while not self.cancel_event.is_set():
                with self.condition:
                    self.condition.wait_for(lambda: self.pending or self.closed, timeout=0.1)
                    if self.closed:
                        break
                    if not self.pending:
                        continue
                # Give other ready workers a chance to enqueue. Never wait for
                # a worker that has finished or takes a different program path.
                if self.cancel_event.wait(0.01):
                    break
                with self.condition:
                    calls, self.pending = self.pending, []
                groups = {}
                for lm, request, future in calls:
                    groups.setdefault(lm, []).append((request, future))
                for lm, group in groups.items():
                    try:
                        if self.cancel_event.is_set():
                            raise LMProviderError("Provider batch execution cancelled")
                        outcomes = lm.provider.batch(
                            lm,
                            [request for request, _ in group],
                            cancel_event=self.cancel_event,
                            timeout=self.timeout,
                            poll_interval=self.poll_interval,
                        )
                        if len(outcomes) != len(group):
                            raise LMProviderError("Provider batch returned an incorrect number of outcomes")
                    except Exception as error:
                        outcomes = [error] * len(group)
                    for (_, future), outcome in zip(group, outcomes, strict=False):
                        if isinstance(outcome, Exception):
                            future.set_exception(outcome)
                        else:
                            future.set_result(outcome)
                calls = []
        finally:
            with self.condition:
                self.closed = True
                remaining = calls + self.pending
                self.pending = []
            for _, _, future in remaining:
                if not future.done():
                    future.set_exception(LMProviderError("Provider batch execution cancelled"))
