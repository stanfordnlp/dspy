import copy
import threading

from contextlib import contextmanager
from contextvars import ContextVar
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
    request_cache=None,
    send_stream=None,
)

# Global base configuration
main_thread_config = copy.deepcopy(DEFAULT_CONFIG)

# Initialize the context variable with an empty dict as default
dspy_ctx_overrides = ContextVar("dspy_ctx_overrides", default=dotdict())


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
        overrides = dspy_ctx_overrides.get()
        if name in overrides:
            return overrides[name]
        elif name in main_thread_config:
            return main_thread_config[name]
        else:
            raise AttributeError(f"'Settings' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in ("_instance",):
            super().__setattr__(name, value)
        else:
            self.configure(**{name: value})

    # Dictionary-like access

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __contains__(self, key):
        overrides = dspy_ctx_overrides.get()
        return key in overrides or key in main_thread_config

    def get(self, key, default=None):
        try:
            return self[key]
        except AttributeError:
            return default

    def copy(self):
        overrides = dspy_ctx_overrides.get()
        return dotdict({**main_thread_config, **overrides})

    @property
    def config(self):
        config = self.copy()
        del config["lock"]
        return config

    # Configuration methods

    def configure(self, return_token=False, **kwargs):
        global main_thread_config
        overrides = dspy_ctx_overrides.get()
        new_overrides = dotdict({**copy.deepcopy(DEFAULT_CONFIG), **main_thread_config, **overrides, **kwargs})
        token = dspy_ctx_overrides.set(new_overrides)

        # Update main_thread_config, in the main thread only
        if threading.current_thread() is threading.main_thread():
            main_thread_config = new_overrides

        if return_token:
            return token

    @contextmanager
    def context(self, **kwargs):
        """Context manager for temporary configuration changes."""
        token = self.configure(return_token=True, **kwargs)
        try:
            yield
        finally:
            dspy_ctx_overrides.reset(token)

            if threading.current_thread() is threading.main_thread():
                global main_thread_config
                main_thread_config = dotdict({**copy.deepcopy(DEFAULT_CONFIG), **dspy_ctx_overrides.get()})

    def __repr__(self):
        overrides = dspy_ctx_overrides.get()
        combined_config = {**main_thread_config, **overrides}
        return repr(combined_config)


settings = Settings()
