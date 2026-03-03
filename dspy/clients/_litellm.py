import logging
import os

_litellm = None


def get_litellm():
    """Import and configure litellm.

    Configuration (telemetry, caching, logging) runs once so that user
    overrides like ``enable_litellm_logging()`` are not clobbered by
    subsequent calls.
    """
    global _litellm
    if _litellm is not None:
        return _litellm

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

    _litellm = litellm
    return _litellm
