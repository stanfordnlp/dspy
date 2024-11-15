import random
from typing import Any, Dict, List, Optional, Union

import dsp
from dspy.predict.parameter import Parameter
from abc import ABC, abstractmethod
from dspy.primitives.prediction import Prediction
from dspy.clients.embedding import Embedder
from dspy.utils.callback import with_callbacks


def single_query_passage(passages):
    passages_dict = {key: [] for key in list(passages[0].keys())}
    for docs in passages:
        for key, value in docs.items():
            passages_dict[key].append(value)
    if "long_text" in passages_dict:
        passages_dict["passages"] = passages_dict.pop("long_text")
    return Prediction(**passages_dict)


class Retrieve(ABC):
    def __init__(self, embedder: Optional[Embedder] = None, k: int = 5, callbacks: Optional[List[Any]] = None):
        self.embedder = embedder
        self.k = k
        self.callbacks = callbacks or []

    @abstractmethod
    def forward(self, query: str, k: Optional[int] = None) -> Any:
        """
        Retrievers implement this method with their custom retrieval logic.
        Must return an object that has a 'passages' attribute (ideally `dspy.Prediction`).
        """
        pass

    def __call__(self, query: str, k: Optional[int] = None) -> Any:
        """
        Calls the forward method and checks if the result has a 'passages' attribute.
        """
        k = k if k is not None else self.k
        result = self.forward(query, k)
        if not hasattr(result, 'passages'):
            raise ValueError("The 'forward' method must return an object with a 'passages' attribute (ideally `dspy.Prediction`).")
        for callback in self.callbacks:
            callback(result)
        return result

# TODO: Consider doing Prediction.from_completions with the individual sets of passages (per query) too.


class RetrieveThenRerank(Parameter):
    name = "Search"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant passages followed by reranking from a corpus"

    def __init__(self, k=3):
        self.stage = random.randbytes(8).hex()
        self.k = k

    def reset(self):
        pass

    def dump_state(self, save_verbose=False):
        """save_verbose is set as a default argument to support the inherited Parameter interface for dump_state"""
        state_keys = ["k"]
        return {k: getattr(self, k) for k in state_keys}

    def load_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        query_or_queries: Union[str, List[str]],
        k: Optional[int] = None,
        with_metadata: bool = False,
        **kwargs,
    ) -> Union[List[str], Prediction, List[Prediction]]:
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [query.strip().split("\n")[0].strip() for query in queries]

        # print(queries)
        # TODO: Consider removing any quote-like markers that surround the query too.
        k = k if k is not None else self.k
        if not with_metadata:
            passages = dsp.retrieveRerankEnsemble(queries, k=k, **kwargs)
            return passages
        else:
            passages = dsp.retrieveRerankEnsemblewithMetadata(queries, k=k, **kwargs)
            if isinstance(passages[0], List):
                pred_returns = []
                for query_passages in passages:
                    passages_dict = {key: [] for key in list(query_passages[0].keys())}
                    for docs in query_passages:
                        for key, value in docs.items():
                            passages_dict[key].append(value)
                    if "long_text" in passages_dict:
                        passages_dict["passages"] = passages_dict.pop("long_text")

                    pred_returns.append(Prediction(**passages_dict))
                return pred_returns
            elif isinstance(passages[0], Dict):
                return single_query_passage(passages=passages)