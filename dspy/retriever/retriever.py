from typing import Any, List, Optional

from abc import ABC, abstractmethod
from dspy.clients.embedding import Embedding
from dspy.utils.callback import with_callbacks

import os
from pathlib import Path
from diskcache import Cache

DISK_CACHE_DIR = os.environ.get("DSPY_CACHEDIR") or os.path.join(Path.home(), ".dspy_cache")


class Retriever(ABC):
    def __init__(self, embedder: Optional[Embedding] = None, k: int = 5, callbacks: Optional[List[Any]] = None, cache: bool = False):
        """
        Interface for composing retrievers in DSPy to return relevant passages or documents based on a query.

        Args:
            embedder (Optional[Embedding]): An instance of `dspy.Embedding` used to compute embeddings
                for queries and documents. If `None`, embedding functionality should be implemented
                within the subclass. Defaults to `None`.
            k (int): The default number of top passages to retrieve when not specified in the `forward` method. Defaults to `5`.
            callbacks (Optional[List[Any]]): A list of callback functions to be called during retrieval.
            cache (bool): Enable retrieval caching. Disabled by default.
        """
        self.embedder = embedder
        self.k = k
        self.callbacks = callbacks or []
        self.cache_enabled = cache
        self.cache = Cache(directory=DISK_CACHE_DIR) if self.cache_enabled else None

    @abstractmethod
    def forward(self, query: str, k: Optional[int] = None) -> Any:
        """
        Retrievers implement this method with their custom retrieval logic.
        Must return an object that has a 'passages' attribute (ideally `dspy.Prediction`).
        """
        pass

    @with_callbacks
    def __call__(self, query: str, k: Optional[int] = None) -> Any:
        """
        Calls the forward method and checks if the result has a 'passages' attribute.
        """
        k = k if k is not None else self.k
        if self.cache_enabled and self.cache is not None:
            cache_key = (query, k)
            try:
                result = self.cache[cache_key]
            except KeyError:
                result = self.forward(query, k)
                self.cache[cache_key] = result
        else:
            result = self.forward(query, k)
        if not hasattr(result, 'passages'):
            raise ValueError(
                "The 'forward' method must return an object with a 'passages' attribute (ideally `dspy.Prediction`)."
            )
        return result
