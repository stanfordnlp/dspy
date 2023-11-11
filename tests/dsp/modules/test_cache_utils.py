import functools
import random

from dsp.modules.cache_utils import cache, set_cache_controller, noop_decorator


def test_noop_cache():
    set_cache_controller(noop_decorator())

    @cache
    def some_func(arg):
        return random.randint(0, 1000000)

    runs = set(some_func(1) for _ in range(1000))

    # no cache would yield different random result each time, but allow some slack
    assert len(runs) > 750


def test_lru_cache():
    set_cache_controller(functools.lru_cache(maxsize=1))

    @cache
    def some_func(arg):
        return random.randint(0, 1000000)

    runs = set(some_func(1) for _ in range(1000))

    # cache denies randomness
    assert len(runs) == 1


def test_joblib_memory():
    import joblib
    import tempfile
    with tempfile.TemporaryDirectory() as dir:
        set_cache_controller(joblib.Memory(location=dir).cache)

        @cache
        def some_func(arg):
            return random.randint(0, 1000000)

        runs = set(some_func(1) for _ in range(1000))

    # cache denies randomness
    assert len(runs) == 1
