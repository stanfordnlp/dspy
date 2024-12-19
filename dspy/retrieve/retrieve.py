from functools import lru_cache
from typing import List, Optional, Union
import logging
import random

import dsp
from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction
from dspy.utils.callback import with_callbacks


def single_query_passage(passages):
    passages_dict = {key: [] for key in list(passages[0].keys())}
    for docs in passages:
        for key, value in docs.items():
            passages_dict[key].append(value)
    if "long_text" in passages_dict:
        passages_dict["passages"] = passages_dict.pop("long_text")
    return Prediction(**passages_dict)


logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def warn_once(msg: str):
    logger.warning(msg)


class Retrieve(Parameter):
    name = "Search"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant passages from a corpus"

    def __init__(self, k=3, callbacks=None):
        warn_once(
            "Existing retriever integrations under dspy/retrieve inheriting `dspy.Retrieve` are deprecated and will be removed in the DSPy 2.7 release. \n"
            "For future retriever integrations, please use the `dspy.Retriever` interface under dspy/retriever/retriever.py and reference any of the custom integrations supported in dspy/retriever/"
        )
        self.stage = random.randbytes(8).hex()
        self.k = k
        self.callbacks = callbacks or []

    def reset(self):
        pass

    def dump_state(self):
        state_keys = ["k"]
        return {k: getattr(self, k) for k in state_keys}

    def load_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    @with_callbacks
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        query: str,
        k: Optional[int] = None,
        **kwargs,
    ) -> Union[List[str], Prediction, List[Prediction]]:
        k = k if k is not None else self.k

        import dspy

        if not dspy.settings.rm:
            raise AssertionError("No RM is loaded.")

        passages = dspy.settings.rm(query, k=k, **kwargs)

        from collections.abc import Iterable

        if not isinstance(passages, Iterable):
            # it's not an iterable yet; make it one.
            # TODO: we should unify the type signatures of dspy.Retriever
            passages = [passages]
        passages = [psg.long_text for psg in passages]

        return Prediction(passages=passages)


# TODO: Consider doing Prediction.from_completions with the individual sets of passages (per query) too.
# TODO potentially add for deprecation/removal in 2.7
class RetrieveThenRerank(Parameter):
    name = "Search"
    input_variable = "query"
    desc = (
        "takes a search query and returns one or more potentially relevant passages followed by reranking from a corpus"
    )

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
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
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
            elif isinstance(passages[0], dict):
                return single_query_passage(passages=passages)
