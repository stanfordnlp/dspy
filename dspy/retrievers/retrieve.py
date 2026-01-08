import random

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


class Retrieve(Parameter):
    """Retrieve passages for a query using the configured retrieval model.

    Uses ``dspy.settings.rm`` (the configured retrieval model) to fetch relevant
    passages given a search query. This module exposes retrieval as a trainable
    DSPy parameter that can be optimized alongside other components.

    Requires a retrieval model to be configured via ``dspy.configure(rm=...)``.

    Args:
        k: Default number of passages to retrieve. Defaults to 3.
        callbacks: Optional list of callbacks invoked during retrieval.

    Attributes:
        name: Display name for the retrieval operation ("Search").
        input_variable: Name of the input variable ("query").
        desc: Description of the retrieval operation.

    Example:
        Retrieve passages for a question:

        ```python
        import dspy

        # Configure a retrieval model (e.g., ColBERTv2)
        rm = dspy.ColBERTv2(url="http://localhost:8893/api/search")
        dspy.configure(rm=rm)

        # Create retriever and fetch passages
        retriever = dspy.Retrieve(k=5)
        result = retriever("What causes rainbows?")
        print(result.passages)
        ```
    """

    name = "Search"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant passages from a corpus"

    def __init__(self, k=3, callbacks=None):
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
        k: int | None = None,
        **kwargs,
    ) -> list[str] | Prediction | list[Prediction]:
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
