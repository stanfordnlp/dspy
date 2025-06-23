import asyncio
import contextvars
import copy
import threading
from contextlib import contextmanager

from dspy.dsp.utils.utils import dotdict

DEFAULT_CONFIG = dotdict(
    lm=None,
    adapter=None,
    rm=None,
    branch_idx=0,
    trace=[],
    callbacks=[],
    async_max_workers=8,
    send_stream=None,
    disable_history=False,
    track_usage=False,
    usage_tracker=None,
    caller_predict=None,
    caller_modules=None,
    stream_listeners=[],
    provide_traceback=False,  # Whether to include traceback information in error logs.
    num_threads=8,  # Number of threads to use for parallel processing.
    max_errors=10,  # Maximum errors before halting operations.
    # If true, async tools can be called in sync mode by getting converted to sync.
    allow_tool_async_sync_conversion=False,
)

# Global base configuration and owner tracking
main_thread_config = copy.deepcopy(DEFAULT_CONFIG)
config_owner_thread_id = None
config_owner_async_task = None

# Global lock for settings configuration
global_lock = threading.Lock()

thread_local_overrides = contextvars.ContextVar("context_overrides", default=dotdict())


class Settings:
    """
    A singleton class for DSPy configuration settings.
    Thread-safe global configuration.
    - 'configure' can be called by only one 'owner' thread (the first thread that calls it).
    - Other threads see the configured global values from 'main_thread_config'.
    - 'context' sets thread-local overrides. These overrides propagate to threads spawned
      inside that context block, when (and only when!) using a ParallelExecutor that copies overrides.

      1. Only one unique thread (which can be any thread!) can call dspy.configure.
      2. It affects a global state, visible to all. As a result, user threads work, but they shouldn't be
         mixed with concurrent changes to dspy.configure from the "main" thread.
         (TODO: In the future, add warnings: if there are near-in-time user-thread reads followed by .configure calls.)
      3. Any thread can use dspy.context. It propagates to child threads created with DSPy primitives: Parallel, asyncify, etc.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def lock(self):
        return global_lock

    def __getattr__(self, name):
        overrides = thread_local_overrides.get()
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

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __contains__(self, key):
        overrides = thread_local_overrides.get()
        return key in overrides or key in main_thread_config

    def get(self, key, default=None):
        try:
            return self[key]
        except AttributeError:
            return default

    def copy(self):
        overrides = thread_local_overrides.get()
        return dotdict({**main_thread_config, **overrides})

    @property
    def config(self):
        return self.copy()

    def _ensure_configure_allowed(self):
        global main_thread_config, config_owner_thread_id, config_owner_async_task
        current_thread_id = threading.get_ident()

        if config_owner_thread_id is None:
            # First `configure` call assigns the owner thread id.
            config_owner_thread_id = current_thread_id

        if config_owner_thread_id != current_thread_id:
            # Disallow a second `configure` calls from other threads.
            raise RuntimeError("dspy.settings can only be changed by the thread that initially configured it.")

        # Async task doesn't allow a second `configure` call, must use dspy.context(...) instead.
        is_async_task = False
        try:
            if asyncio.current_task() is not None:
                is_async_task = True
        except RuntimeError:
            # This exception (e.g., "no current task") means we are not in an async loop/task,
            # or asyncio module itself is not fully functional in this specific sub-thread context.
            is_async_task = False

        if not is_async_task:
            return

        if config_owner_async_task is None:
            # First `configure` call assigns the owner async task.
            config_owner_async_task = asyncio.current_task()
            return

        # We are in an async task. Now check for IPython and allow calling `configure` from IPython.
        in_ipython = False
        try:
            from IPython import get_ipython

            # get_ipython is a global injected by IPython environments.
            # We check its existence and type to be more robust.
            in_ipython = get_ipython() is not None
        except Exception:
            # If `IPython` is not installed or `get_ipython` failed, we are not in an IPython environment.
            in_ipython = False

        if not in_ipython and config_owner_async_task != asyncio.current_task():
            raise RuntimeError(
                "dspy.settings.configure(...) can only be called from the same async task that called it first. Please "
                "use `dspy.context(...)` in other async tasks instead."
            )

    def configure(self, **kwargs):
        # If no exception is raised, the `configure` call is allowed.
        self._ensure_configure_allowed()

        # Update global config
        for k, v in kwargs.items():
            main_thread_config[k] = v

    @contextmanager
    def context(self, **kwargs):
        """
        Context manager for temporary configuration changes at the thread level.
        Does not affect global configuration. Changes only apply to the current thread.
        If threads are spawned inside this block using ParallelExecutor, they will inherit these overrides.
        """

        original_overrides = thread_local_overrides.get().copy()
        new_overrides = dotdict({**main_thread_config, **original_overrides, **kwargs})
        token = thread_local_overrides.set(new_overrides)

        try:
            yield
        finally:
            thread_local_overrides.reset(token)

    def __repr__(self):
        overrides = thread_local_overrides.get()
        combined_config = {**main_thread_config, **overrides}
        return repr(combined_config)


settings = Settings()
