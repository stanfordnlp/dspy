"""One-time litellm configuration.

Importing this module loads and configures litellm.  Python's import
system guarantees the module body runs exactly once, so no flag or
guard is needed.
"""

import logging

import litellm
from litellm._logging import verbose_logger

litellm.telemetry = False
litellm.cache = None
litellm.suppress_debug_info = True

verbose_logger.setLevel(logging.ERROR)
for h in verbose_logger.handlers:
    h.setLevel(logging.ERROR)
