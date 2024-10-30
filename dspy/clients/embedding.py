import os
from pathlib import Path

import litellm
from litellm.caching import Cache

DISK_CACHE_DIR = os.environ.get("DSPY_CACHEDIR") or os.path.join(Path.home(), ".dspy_cache")
litellm.cache = Cache(disk_cache_dir=DISK_CACHE_DIR, type="disk")
litellm.telemetry = False

if "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ:
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


class Embedding:
    def __init__(self, model):
        self.model = model

    def __call__(self, inputs, caching=True, **kwargs):
        if isinstance(self.model, str):
            return litellm.embedding(model=self.model, input=inputs, caching=caching, **kwargs)
        elif isinstance(self.model, callable):
            return self.model(inputs, **kwargs)
        else:
            raise ValueError(f"`model` in `dspy.Embedding` must be a string or a callable, but got {type(self.model)}.")
