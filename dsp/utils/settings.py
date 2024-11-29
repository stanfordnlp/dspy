import copy
import threading
from contextlib import contextmanager
from dsp.utils.utils import dotdict

DEFAULT_CONFIG = dotdict(
    lm=None,
    adapter=None,
    rm=None,
    branch_idx=0,
    reranker=None,
    compiled_lm=None,
    force_reuse_cached_compilation=False,
    compiling=False,
    skip_logprobs=False,
    trace=[],
    release=0,
    bypass_assert=False,
    bypass_suggest=False,
    assert_failures=0,
    suggest_failures=0,
    langchain_history=[],
    experimental=False,
    backoff_time=10,
    callbacks=[],
    async_max_workers=8,
)

# Global base configuration
main_thread_config = copy.deepcopy(DEFAULT_CONFIG)


class ThreadLocalOverrides(threading.local):
    def __init__(self):
        self.overrides = dotdict()  # Initialize thread-local overrides


# Create the thread-local storage
thread_local_overrides = ThreadLocalOverrides()


class Settings:
    """
    A singleton class for DSPy configuration settings.

    This is thread-safe. User threads are supported both through ParallelExecutor and native threading.
        - If native threading is used, the thread inherits the initial config from the main thread.
        - If ParallelExecutor is used, the thread inherits the initial config from its parent thread.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.lock = threading.Lock()  # maintained here for DSPy assertions.py
        return cls._instance

    def __getattr__(self, name):
        overrides = getattr(thread_local_overrides, 'overrides', dotdict())
        if name in overrides:
            return overrides[name]
        elif name in main_thread_config:
            return main_thread_config[name]
        else:
            raise AttributeError(f"'Settings' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in ('_instance',):
            super().__setattr__(name, value)
        else:
            self.configure(**{name: value})

    # Dictionary-like access

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __contains__(self, key):
        overrides = getattr(thread_local_overrides, 'overrides', dotdict())
        return key in overrides or key in main_thread_config

    def get(self, key, default=None):
        try:
            return self[key]
        except AttributeError:
            return default

    def copy(self):
        overrides = getattr(thread_local_overrides, 'overrides', dotdict())
        return dotdict({**main_thread_config, **overrides})

    @property
    def config(self):
        config = self.copy()
        if 'lock' in config:
            del config['lock']
        return config

    # Configuration methods

    def configure(self, **kwargs):
        global main_thread_config

        # Get or initialize thread-local overrides
        overrides = getattr(thread_local_overrides, 'overrides', dotdict())
        thread_local_overrides.overrides = dotdict(
            {**copy.deepcopy(DEFAULT_CONFIG), **main_thread_config, **overrides, **kwargs}
        )

        # Update main_thread_config, in the main thread only
        if threading.current_thread() is threading.main_thread():
            main_thread_config = thread_local_overrides.overrides

    @contextmanager
    def context(self, **kwargs):
        """Context manager for temporary configuration changes."""
        global main_thread_config
        original_overrides = getattr(thread_local_overrides, 'overrides', dotdict()).copy()
        original_main_thread_config = main_thread_config.copy()

        self.configure(**kwargs)
        try:
            yield
        finally:
            thread_local_overrides.overrides = original_overrides

            if threading.current_thread() is threading.main_thread():
                main_thread_config = original_main_thread_config

    def __repr__(self):
        overrides = getattr(thread_local_overrides, 'overrides', dotdict())
        combined_config = {**main_thread_config, **overrides}
        return repr(combined_config)


settings = Settings()
