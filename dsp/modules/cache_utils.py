import os
from functools import wraps
from pathlib import Path

from joblib import Memory

from dsp.utils import dotdict

cache_turn_on = not os.environ.get('DSP_CACHEBOOL', 'True').lower() == 'false'


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


cachedir = os.environ.get('DSP_CACHEDIR') or os.path.join(Path.home(), 'cachedir_joblib')
CacheMemory = Memory(location=cachedir, verbose=0)

cachedir2 = os.environ.get('DSP_NOTEBOOK_CACHEDIR')
NotebookCacheMemory = dotdict()
NotebookCacheMemory.cache = noop_decorator

if cachedir2:
    NotebookCacheMemory = Memory(location=cachedir2, verbose=0)


if not cache_turn_on:
    CacheMemory = dotdict()
    CacheMemory.cache = noop_decorator

    NotebookCacheMemory = dotdict()
    NotebookCacheMemory.cache = noop_decorator
