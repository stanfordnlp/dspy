from .lm import LM
from .provider import Provider, TrainingJob
from .base_lm import BaseLM, inspect_history
from .embedding import Embedding
import litellm
import os
from pathlib import Path
from litellm.caching import Cache

DISK_CACHE_DIR = os.environ.get("DSPY_CACHEDIR") or os.path.join(Path.home(), ".dspy_cache")
litellm.cache = Cache(disk_cache_dir=DISK_CACHE_DIR, type="disk")
litellm.telemetry = False
# Turn off by default to avoid LiteLLM logging during every LM call.
litellm.suppress_debug_info = True

if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
    # Accessed at run time by litellm; i.e., fine to keep after import
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

def enable_litellm_logging():
    litellm.suppress_debug_info = False

def disable_litellm_logging():
    litellm.suppress_debug_info = True
