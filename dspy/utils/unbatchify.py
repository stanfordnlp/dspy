import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable, List


class Unbatchify:
    def __init__(
        self,
        batch_fn: Callable[[List[Any]], List[Any]],
        max_batch_size: int = 32,
        max_wait_time: float = 0.1
    ):
        """
        Initializes the Unbatchify.

        Args:
            batch_fn: The batch-processing function that accepts a list of inputs and returns a list of outputs.
            max_batch_size: The maximum number of items to include in a batch.
            max_wait_time: The maximum time (in seconds) to wait for batch to fill before processing.
        """

        self.batch_fn = batch_fn
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.input_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._worker)
        self.worker_thread.daemon = True  # Ensures thread exits when main program exits
        self.worker_thread.start()

    def __call__(self, input_item: Any) -> Any:
        """
        Thread-safe function that accepts a single input and returns the corresponding output.

        Args:
            input_item: The single input item to process.

        Returns:
            The output corresponding to the input_item after processing through batch_fn.
        """
        future = Future()
        self.input_queue.put((input_item, future))
        try:
            result = future.result()
        except Exception as e:
            raise e
        return result

    def _worker(self):
        """
        Worker thread that batches inputs and processes them using batch_fn.
        """
        while not self.stop_event.is_set():
            batch = []
            futures = []
            start_time = time.time()
            while len(batch) < self.max_batch_size and (time.time() - start_time) < self.max_wait_time:
                try:
                    input_item, future = self.input_queue.get(timeout=self.max_wait_time)
                    batch.append(input_item)
                    futures.append(future)
                except queue.Empty:
                    break

            if batch:
                try:
                    outputs = self.batch_fn(batch)
                    for output, future in zip(outputs, futures):
                        future.set_result(output)
                except Exception as e:
                    for future in futures:
                        future.set_exception(e)
            else:
                time.sleep(0.01)

        # Clean up remaining items when stopping
        while True:
            try:
                _, future = self.input_queue.get_nowait()
                future.set_exception(RuntimeError("Unbatchify is closed"))
            except queue.Empty:
                break

        print("Worker thread has been terminated.")

    def close(self):
        """
        Stops the worker thread and cleans up resources.
        """
        if not self.stop_event.is_set():
            self.stop_event.set()
            self.worker_thread.join()

    def __enter__(self):
        """
        Enables use as a context manager.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Ensures resources are cleaned up when exiting context.
        """
        self.close()

    def __del__(self):
        """
        Ensures the worker thread is terminated when the object is garbage collected.
        """
        self.close()
