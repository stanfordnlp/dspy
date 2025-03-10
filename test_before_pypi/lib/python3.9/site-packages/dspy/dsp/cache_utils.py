import os
from functools import wraps
from pathlib import Path

from joblib import Memory

from dspy.dsp.utils import dotdict

cache_turn_on = os.environ.get('DSP_CACHEBOOL', 'True').lower() != 'false'


def noop_decorator(arg=None, *noop_args, **noop_kwargs):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    if callable(arg):
        return decorator(arg)
    else:
        return decorator

NotebookCacheMemory = dotdict()
NotebookCacheMemory.cache = noop_decorator

CacheMemory = dotdict()
CacheMemory.cache = noop_decorator

if cache_turn_on:
    cachedir = os.environ.get('DSP_CACHEDIR') or os.path.join(Path.home(), '.dspy_cache/cachedir_joblib')
    CacheMemory = Memory(location=cachedir, verbose=0)

    cachedir2 = os.environ.get('DSP_NOTEBOOK_CACHEDIR')
    if cachedir2:
        NotebookCacheMemory = Memory(location=cachedir2, verbose=0)
