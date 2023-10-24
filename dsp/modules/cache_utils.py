import functools
import os
from functools import wraps

__all__ = (
    'cache',
    'set_cache_controller'
)


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


def set_cache_controller(obj: callable):
    """
    Caching protocol:
    DSPy provides hook for caching, but does not implement the actual caching logic, as this is a responsibility of
    end-users, and not of a foundational library.
    DSPy hence conforms only to a cache protocol that matches that of 'functools.lru_cache', that is a master callable
    that takes as arguments the cache properties.
    It then produces another callable, the decorator, whose sole argument is the target function to cache.
    The enclosed target functions' arguments then can be used in any combination to determine unique cache keys.

    Example 1
    ---------
    # Configure the most basic cache system possible
    set_cache_controller(lambda func: functools.lru_cache(maxsize=1000)(func))

    Example 2
    ---------
    # Slightly more advanced cache, with disk persistence
    set_cache_controller(joblib.Memory(location=...).cache)

    Example 3
    ---------
    # some code defined by the end-user could be
    class RedisCache:
        # mock code for a cache backed by Redis
        def __init__(host): ...
        def __call__(func):
            @functools.wraps(func)
            def decorator(*args, **kwargs):
                # compute cache key from *args and **kwargs
                key = ...
                if key not in redis:
                    res = func(*args, **kwargs)
                    redis.set(key, res)
                else:
                    res = redis.get(key)
                return res
            return decorator

    # then set the cache system
    set_cache_controller(RedisCache(host=...))
    """
    global cache
    cache = obj


__cachedir = (
        os.environ.get('DSP_NOTEBOOK_CACHEDIR')
        or os.environ.get('DSP_CACHEDIR')
)
try:
    __cachesize = int(os.environ.get('DSP_CACHESIZE'))
except (TypeError, ValueError):
    __cachesize = None
if __cachedir is not None:
    # kept for backward compatibility only
    from joblib import Memory

    cache = Memory(location=__cachedir, verbose=0)
    cache.reduce_size(__cachesize)
    cache = cache.cache
elif __cachesize is not None:
    # kept for backward compatibility only
    cache = functools.lru_cache(maxsize=__cachesize)
else:
    cache = noop_decorator()
