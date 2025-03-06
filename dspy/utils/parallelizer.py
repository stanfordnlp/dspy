import sys
import tqdm
import signal
import logging
import threading
import traceback
import time
import contextlib

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

logger = logging.getLogger(__name__)


class ParallelExecutor:
    def __init__(
        self,
        num_threads,
        max_errors=5,
        disable_progress_bar=False,
        provide_traceback=False,
        compare_results=False,
        timeout=120,
        straggler_limit=3,
    ):
        """
        Offers isolation between the tasks (dspy.settings) irrespective of whether num_threads == 1 or > 1.
        Handles also straggler timeouts.
        """
        
        self.num_threads = num_threads
        self.max_errors = max_errors
        self.disable_progress_bar = disable_progress_bar
        self.provide_traceback = provide_traceback
        self.compare_results = compare_results
        self.timeout = timeout
        self.straggler_limit = straggler_limit

        self.error_count = 0
        self.error_lock = threading.Lock()
        self.cancel_jobs = threading.Event()

    def execute(self, function, data):
        wrapped = self._wrap_function(function)
        return self._execute_parallel(wrapped, data)

    def _wrap_function(self, user_function):
        def safe_func(item):
            if self.cancel_jobs.is_set():
                return None
            try:
                return user_function(item)
            except Exception as e:
                with self.error_lock:
                    self.error_count += 1
                    if self.error_count >= self.max_errors:
                        self.cancel_jobs.set()
                if self.provide_traceback:
                    logger.error(f"Error for {item}: {e}\n{traceback.format_exc()}")
                else:
                    logger.error(
                        f"Error for {item}: {e}. "
                        "Set `provide_traceback=True` for traceback."
                    )
                return None

        return safe_func

    def _execute_parallel(self, function, data):
        results = [None] * len(data)
        job_cancelled = "cancelled"

        # We resubmit at most once per item.
        start_time_map = {}
        start_time_lock = threading.Lock()
        resubmitted = set()

        # This is the worker function each thread will run.
        def worker(parent_overrides, submission_id, index, item):
            if self.cancel_jobs.is_set():
                return index, job_cancelled
            # Record actual start time
            with start_time_lock:
                start_time_map[submission_id] = time.time()

            # Apply parent's thread-local overrides
            from dspy.dsp.utils.settings import thread_local_overrides

            original = thread_local_overrides.overrides
            thread_local_overrides.overrides = parent_overrides.copy()

            try:
                return index, function(item)
            finally:
                thread_local_overrides.overrides = original

        # Handle Ctrl-C in the main thread
        @contextlib.contextmanager
        def interrupt_manager():
            if threading.current_thread() is threading.main_thread():
                orig_handler = signal.getsignal(signal.SIGINT)

                def handler(sig, frame):
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
                from dspy.dsp.utils.settings import thread_local_overrides

                parent_overrides = thread_local_overrides.overrides.copy()

                futures_map = {}
                futures_set = set()
                submission_counter = 0

                for idx, item in enumerate(data):
                    f = executor.submit(
                        worker, parent_overrides, submission_counter, idx, item
                    )
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
                    return all(r is not None for r in results)

                while futures_set and not self.cancel_jobs.is_set():
                    if all_done():
                        break
                    done, not_done = wait(
                        futures_set, timeout=1, return_when=FIRST_COMPLETED
                    )
                    for f in done:
                        futures_set.remove(f)
                        try:
                            index, outcome = f.result()
                        except Exception:
                            pass
                        else:
                            if outcome != job_cancelled and results[index] is None:
                                results[index] = outcome

                            # Update progress
                            if self.compare_results:
                                vals = [r[-1] for r in results if r is not None]
                                self._update_progress(pbar, sum(vals), len(vals))
                            else:
                                self._update_progress(
                                    pbar,
                                    len([r for r in results if r is not None]),
                                    len(data),
                                )

                    if all_done():
                        break

                    # Check stragglers if few remain
                    if 0 < self.timeout and len(not_done) <= self.straggler_limit:
                        now = time.time()
                        for f in list(not_done):
                            if f not in resubmitted:
                                sid, idx, item = futures_map[f]
                                with start_time_lock:
                                    st = start_time_map.get(sid, None)
                                if st and (now - st) >= self.timeout:
                                    resubmitted.add(f)
                                    nf = executor.submit(
                                        worker,
                                        parent_overrides,
                                        submission_counter,
                                        idx,
                                        item,
                                    )
                                    futures_map[nf] = (submission_counter, idx, item)
                                    futures_set.add(nf)
                                    submission_counter += 1

                pbar.close()

        finally:
            # Avoid waiting on leftover tasks that no longer matter
            executor.shutdown(wait=False)

        if self.cancel_jobs.is_set():
            logger.warning("Execution cancelled due to errors or interruption.")
            raise Exception("Execution cancelled due to errors or interruption.")

        return results

    def _update_progress(self, pbar, nresults, ntotal):
        if self.compare_results:
            pct = round(100 * nresults / ntotal, 1) if ntotal else 0
            pbar.set_description(f"Average Metric: {nresults:.2f} / {ntotal} ({pct}%)")
        else:
            pbar.set_description(f"Processed {nresults} / {ntotal} examples")
        pbar.update()
