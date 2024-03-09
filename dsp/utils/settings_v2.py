import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager


class Settings:
    def __init__(self):
        # A lock for ensuring thread-safety when accessing _parent_configs
        self._lock = threading.Lock()
        
        # Dictionary to hold parent thread configurations
        self._parent_configs = {}
        
        # Using thread-local storage to ensure that each thread has its own configuration stack
        self._local = threading.local()

    def _get_current_config(self):
        return self._local.config_stack[-1] if hasattr(self._local, 'config_stack') and self._local.config_stack else {}

    def initialize_for_thread(self, parent_tid):
        """Initialize thread-local data for a new thread using its parent's config."""
        with self._lock:
            parent_config = self._parent_configs.get(parent_tid)
            if parent_config:
                self._local.config_stack = [copy.deepcopy(parent_config)]
            else:
                self._local.config_stack = [{}]

    @contextmanager
    def context(self, **kwargs):
        current_config = copy.deepcopy(self._get_current_config())  # Deep copy the current configuration
        current_config.update(kwargs)
        
        if not hasattr(self._local, 'config_stack'):
            self._local.config_stack = []
        
        self._local.config_stack.append(current_config)

        # Register the modified config as the potential parent config
        with self._lock:
            self._parent_configs[threading.get_ident()] = copy.deepcopy(current_config)  # Deep copy to ensure immutability

        try:
            yield
        finally:
            self._local.config_stack.pop()

            # Cleanup after exiting the context
            with self._lock:
                self._parent_configs.pop(threading.get_ident(), None)

# Singleton instance
dsp_settings = Settings()


# Wrapper for ThreadPoolExecutor usage
def thread_wrapper(program, parent_tid, *args, **kwargs):
    dsp_settings.initialize_for_thread(parent_tid)
    return program(*args, **kwargs)


# Example test
def sample_program(arg):
    print(f"Thread {threading.get_ident()} with arg={arg} has config: {dsp_settings._get_current_config()}")


def main():
    parent_tid = threading.get_ident()

    with dsp_settings.context(a=10, b=20):  # Setting main thread's context
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {executor.submit(thread_wrapper, sample_program, parent_tid, arg) for arg in range(3)}

            for future in as_completed(futures):
                res = future.result()

        print(f"Main thread {parent_tid} config after threads: {dsp_settings._get_current_config()}")


if __name__ == "__main__":
    main()
