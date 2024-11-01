from typing import Any, Callable, List, Optional

from dspy.primitives.prediction import Prediction
from dspy.retrieve.embedder import Embedder

class RM:
    def __init__(
        self,
        search_function: Callable[..., Any],
        embedder: Optional[Embedder] = None,
        result_formatter: Optional[Callable[[Any], Prediction]] = None,
        **provider_kwargs
    ):
        self.embedder = embedder
        self.search_function = search_function
        self.result_formatter = result_formatter or self.default_formatter
        self.provider_kwargs = provider_kwargs

    def __call__(self, query: str, k: Optional[int] = None) -> Prediction:
        if self.embedder:
            query_vector = self.embedder([query])[0]
            query_input = query_vector
        else:
            query_input = query
        search_args = self.provider_kwargs.copy()
        search_args['query'] = query_input
        if k is not None:
            search_args['k'] = k
        results = self.search_function(**search_args)
        return self.result_formatter(results)

    def default_formatter(self, results) -> Prediction:
        return Prediction(passages=results)