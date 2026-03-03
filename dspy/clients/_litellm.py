import functools
import logging
import os


@functools.lru_cache(maxsize=1)
def get_litellm():
    """Import and configure litellm.

    The first call imports litellm and applies DSPy's default configuration
    (telemetry off, caching off, debug logging silenced). Subsequent calls
    return the already-configured module instantly via lru_cache.
    """
    # Use litellm's bundled model cost map instead of fetching from GitHub
    # on every import (~110ms saved). Must be set before importing litellm.
    if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

    import litellm
    from litellm._logging import verbose_logger

    litellm.telemetry = False
    litellm.cache = None
    litellm.suppress_debug_info = True

    verbose_logger.setLevel(logging.ERROR)
    for h in verbose_logger.handlers:
        h.setLevel(logging.ERROR)

    return litellm
