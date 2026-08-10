import contextlib
import copy
import logging
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import tqdm

from dspy.dsp.utils.settings import settings, thread_local_overrides
from dspy.utils.callback_context import _bind_active_call_id

logger = logging.getLogger(__name__)

_UNSET = object()


class ParallelExecutor:
    def __init__(
        self,
        num_threads=None,
        max_errors=None,
        disable_progress_bar=False,
        provide_traceback=None,
        compare_results=False,
        timeout=120,
        straggler_limit=3,
    ):
        """
        Propagates DSPy settings and callback ancestry into each task while isolating task-local changes,
        irrespective of whether num_threads == 1 or > 1. Handles also straggler timeouts.

        Cancellation is cooperative: execute() stops waiting and returns/raises, but a task
        already running on a pool thread cannot be killed -- it keeps running in the background
        until it finishes, and may delay interpreter exit if it blocks indefinitely.
        """
        self.num_threads = num_threads or settings.num_threads
        self.max_errors = settings.max_errors if max_errors is None else max_errors
        self.disable_progress_bar = disable_progress_bar
        self.provide_traceback = provide_traceback if provide_traceback is not None else settings.provide_traceback
        self.compare_results = compare_results
        self.timeout = timeout
        self.straggler_limit = straggler_limit

        self.error_count = 0
        self.error_lock = threading.Lock()
        self.cancel_jobs = threading.Event()
        self.interrupted = threading.Event()
        self.failed_indices = []
        self.exceptions_map = {}

    def execute(self, function, data):
        tqdm.tqdm._instances.clear()
        wrapped = self._wrap_function(_bind_active_call_id(function))
        if self.num_threads == 1:
            return self._execute_sequential(wrapped, data)
        return self._execute_parallel(wrapped, data)

    def _record_error(self, item, e):
        with self.error_lock:
            self.error_count += 1
            if self.error_count >= self.max_errors:
                self.cancel_jobs.set()
        if self.provide_traceback:
            logger.error(f"Error for {item}: {e}\n{traceback.format_exc()}")
        else:
            logger.error(f"Error for {item}: {e}. Set `provide_traceback=True` for traceback.")

    def _wrap_function(self, user_function):
        def safe_func(item):
            try:
                return user_function(item)
            except Exception as e:
                # recorded by the main thread once the outcome is accepted, so a
                # straggler's duplicate failure can't consume the error budget twice
                return e

        return safe_func

    def _execute_sequential(self, function, data):
        """Execute items sequentially on the main thread."""
        results = [_UNSET] * len(data)

        pbar = tqdm.tqdm(
            total=len(data),
            dynamic_ncols=True,
            disable=self.disable_progress_bar,
            file=sys.stdout,
        )

        try:
            for idx, item in enumerate(data):
                if self.cancel_jobs.is_set():
                    break

                outcome = function(item)
                self._process_outcome(results, idx, outcome)
                if isinstance(outcome, Exception):
                    self._record_error(item, outcome)
                self._report_progress(pbar, results, len(data))
        except KeyboardInterrupt:
            self.interrupted.set()
            self.cancel_jobs.set()
            logger.warning("SIGINT received. Cancelling.")
            raise
        finally:
            pbar.close()

        if self.cancel_jobs.is_set():
            logger.warning("Execution cancelled due to errors or interruption.")
            raise Exception("Execution cancelled due to errors or interruption.")

        return [None if result is _UNSET else result for result in results]

    def _execute_parallel(self, function, data):
        results = [_UNSET] * len(data)
        job_cancelled = "cancelled"

        # We resubmit at most once per logical index.
        start_time_map = {}
        start_time_lock = threading.Lock()
        resubmitted = set()
        skipped = []

        # This is the worker function each thread will run.
        def worker(parent_overrides, submission_id, index, item):
            if self.cancel_jobs.is_set():
                if self.interrupted.is_set() or index not in self.exceptions_map:
                    return index, job_cancelled
            # Record actual start time
            with start_time_lock:
                start_time_map[submission_id] = time.time()

            # Apply parent's thread-local overrides
            original = thread_local_overrides.get()
            new_overrides = {**original, **parent_overrides.copy()}
            if new_overrides.get("usage_tracker"):
                # Usage tracker needs to be deep copied across threads so that each thread tracks its own usage
                new_overrides["usage_tracker"] = copy.deepcopy(new_overrides["usage_tracker"])
            token = thread_local_overrides.set(new_overrides)

            try:
                return index, function(item)
            finally:
                thread_local_overrides.reset(token)

        # Handle Ctrl-C in the main thread
        @contextlib.contextmanager
        def interrupt_manager():
            if threading.current_thread() is threading.main_thread():
                orig_handler = signal.getsignal(signal.SIGINT)

                def handler(sig, frame):
                    self.interrupted.set()
                    self.cancel_jobs.set()
                    logger.warning("SIGINT received. Cancelling.")
                    orig_handler(sig, frame)

                signal.signal(signal.SIGINT, handler)
                try:
                    yield
                finally:
                    signal.signal(signal.SIGINT, orig_handler)
            else:
                yield

        executor = ThreadPoolExecutor(max_workers=self.num_threads)
        try:
            with interrupt_manager():
                parent_overrides = thread_local_overrides.get().copy()

                futures_map = {}
                futures_set = set()
                submission_counter = 0

                for idx, item in enumerate(data):
                    f = executor.submit(worker, parent_overrides, submission_counter, idx, item)
                    futures_map[f] = (submission_counter, idx, item)
                    futures_set.add(f)
                    submission_counter += 1

                pbar = tqdm.tqdm(
                    total=len(data),
                    dynamic_ncols=True,
                    disable=self.disable_progress_bar,
                    file=sys.stdout,
                )

                def all_done():
                    return all(result is not _UNSET for result in results)

                def submit(idx, item):
                    nonlocal submission_counter
                    nf = executor.submit(worker, parent_overrides, submission_counter, idx, item)
                    futures_map[nf] = (submission_counter, idx, item)
                    futures_set.add(nf)
                    submission_counter += 1

                recovery_grace = None

                def keep_running():
                    # After budget cancellation, in-flight retries for failed indices get one
                    # straggler-timeout grace period to recover; then they are abandoned
                    # explicitly and the cancellation stands. A real interrupt exits at once.
                    nonlocal recovery_grace
                    if not self.cancel_jobs.is_set():
                        recovery_grace = None
                        return True
                    if self.interrupted.is_set():
                        return False
                    recoverable = [f for f in futures_set if futures_map[f][1] in self.exceptions_map]
                    if not recoverable:
                        return False
                    recovery_grace = recovery_grace or time.time() + max(self.timeout, 1.0)
                    if time.time() < recovery_grace:
                        return True
                    for f in recoverable:
                        futures_set.discard(f)
                        logger.warning(
                            f"Abandoning in-flight retry for {futures_map[f][2]} after the "
                            "cancellation grace period; its recorded failure stands. The retry "
                            "cannot be killed and keeps running on its pool thread until it "
                            "finishes; if it blocks indefinitely it may delay interpreter exit."
                        )
                    return False

                while futures_set and keep_running():
                    if all_done():
                        break
                    done, not_done = wait(futures_set, timeout=1, return_when=FIRST_COMPLETED)
                    for f in done:
                        futures_set.remove(f)
                        try:
                            index, outcome = f.result()
                        except Exception as e:
                            # A harness-internal failure (e.g. copying thread-local overrides), not a
                            # user-function exception — record it the same way so it isn't silently dropped.
                            _, idx, item = futures_map[f]
                            if self._should_finalize(idx, e, results):
                                self._process_outcome(results, idx, e)
                                self._record_error(item, e)
                            self._report_progress(pbar, results, len(data))
                        else:
                            if outcome == job_cancelled:
                                skipped.append((index, futures_map[f][2]))
                            elif self._should_finalize(index, outcome, results):
                                if isinstance(outcome, Exception):
                                    self._record_error(futures_map[f][2], outcome)
                                else:
                                    # The retry (or the original, if this is the retry) recovered after
                                    # the other future for this index failed -- a success always wins,
                                    # so drop the stale failure record instead of reporting both.
                                    self._clear_stale_failure(index)
                                self._process_outcome(results, index, outcome)

                            self._report_progress(pbar, results, len(data))

                    if all_done():
                        break

                    # Check stragglers if few remain
                    if 0 < self.timeout and len(not_done) <= self.straggler_limit:
                        now = time.time()
                        for f in list(not_done):
                            sid, idx, item = futures_map[f]
                            if idx not in resubmitted:
                                with start_time_lock:
                                    st = start_time_map.get(sid, None)
                                if st and (now - st) >= self.timeout:
                                    resubmitted.add(idx)
                                    submit(idx, item)

                    # Items skipped by the worker gate during a since-revoked cancellation
                    # would otherwise end as silent Nones -- resubmit them.
                    if skipped and not self.cancel_jobs.is_set():
                        for idx, item in skipped:
                            if results[idx] is _UNSET and idx not in self.exceptions_map:
                                submit(idx, item)
                        skipped.clear()

                pbar.close()

        finally:
            # Avoid waiting on leftover tasks that no longer matter; queued-but-unstarted
            # ones are cancelled outright so they never run after execute() returns.
            executor.shutdown(wait=False, cancel_futures=True)

        if self.cancel_jobs.is_set():
            logger.warning("Execution cancelled due to errors or interruption.")
            raise Exception("Execution cancelled due to errors or interruption.")

        return [None if result is _UNSET else result for result in results]

    def _process_outcome(self, results, idx, outcome):
        """Store a single outcome and track errors."""
        if isinstance(outcome, Exception):
            with self.error_lock:
                self.failed_indices.append(idx)
                self.exceptions_map[idx] = outcome
        else:
            results[idx] = outcome

    def _should_finalize(self, idx, outcome, results):
        """Whether this future's outcome should be recorded for idx, given what the other
        future for the same straggler-retried index may have already recorded.

        results[idx] stays _UNSET for an exception outcome (see _process_outcome), so
        checking against _UNSET -- not None, which a task can legitimately return -- is
        what lets this tell "not yet recorded" apart from "already recorded by the other
        future" once both completions are failures. A later success always overrides an
        earlier failure (the retry recovered); a later failure is dropped once idx already
        has an outcome either way, so one logical input can't be double-counted or have its
        result silently overwritten by a duplicate completion.
        """
        if results[idx] is not _UNSET:
            return False
        if isinstance(outcome, Exception):
            return idx not in self.exceptions_map
        return True

    def _clear_stale_failure(self, idx):
        """A retry recovered idx after its original attempt failed. Undo what _record_error
        did for that stale failure: give back error_count, and cancel_jobs too if it was the
        error budget (not a real interrupt) that set it."""
        with self.error_lock:
            if self.exceptions_map.pop(idx, None) is not None:
                self.error_count -= 1
                if not self.interrupted.is_set() and self.error_count < self.max_errors:
                    self.cancel_jobs.clear()
            if idx in self.failed_indices:
                self.failed_indices.remove(idx)

    def _report_progress(self, pbar, results, total):
        """Compute metrics and update the progress bar."""
        if self.compare_results:
            vals = [result[-1] for result in results if result is not _UNSET and result is not None]
            self._update_progress(pbar, sum(vals), len(vals))
        else:
            self._update_progress(
                pbar,
                len([result for result in results if result is not _UNSET]),
                total,
            )

    def _update_progress(self, pbar, nresults, ntotal):
        if self.compare_results:
            pct = round(100 * nresults / ntotal, 1) if ntotal else 0
            pbar.set_description(f"Average Metric: {nresults:.2f} / {ntotal} ({pct}%)")
        else:
            pbar.set_description(f"Processed {nresults} / {ntotal} examples")
        pbar.update()
