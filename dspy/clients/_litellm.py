import functools
import logging


@functools.lru_cache(maxsize=1)
def get_litellm():
    """Import and configure litellm.

    The first call imports litellm and applies DSPy's default configuration
    (telemetry off, caching off, debug logging silenced). Subsequent calls
    return the already-configured module instantly via lru_cache.
    """
    import litellm
    from litellm._logging import verbose_logger

    litellm.telemetry = False
    litellm.cache = None
    litellm.suppress_debug_info = True

    verbose_logger.setLevel(logging.ERROR)
    for h in verbose_logger.handlers:
        h.setLevel(logging.ERROR)

    return litellm
