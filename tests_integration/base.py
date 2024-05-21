import os
from pathlib import Path

import pytest


class BaseIntegrationTestWithCache:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        # Base directory for all DSPy modules
        library_dir = Path(__file__).resolve().parent
        base_dir = library_dir.parent
        cache_dir = str(base_dir / "cache")
        os.environ["DSP_NOTEBOOK_CACHEDIR"] = cache_dir

        if cache_dir and not Path(cache_dir).exists():
            Path(cache_dir).mkdir(parents=True)
