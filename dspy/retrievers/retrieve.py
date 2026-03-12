import random

from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction
from dspy.utils.callback import with_callbacks


def single_query_passage(passages):
    """Pack retriever results for one query into a single prediction.

    The input is expected to be a non-empty sequence of mapping-like passage records
    that all share the same keys. Values for each key are collected into lists. If a
    ``long_text`` key is present, it is renamed to ``passages`` to match the default
    retrieval output shape used throughout DSPy.

    Args:
        passages: Passage records returned for a single query.

    Returns:
        Prediction: A prediction whose fields contain the collected passage values.

    Examples:
        ```python
        docs = [
            {"long_text": "Ada Lovelace wrote notes on the Analytical Engine.", "score": 0.9},
            {"long_text": "Charles Babbage designed the Analytical Engine.", "score": 0.8},
        ]
        result = single_query_passage(docs)

        assert result.passages[0].startswith("Ada Lovelace")
        assert result.score == [0.9, 0.8]
        ```
    """
    passages_dict = {key: [] for key in list(passages[0].keys())}
    for docs in passages:
        for key, value in docs.items():
            passages_dict[key].append(value)
    if "long_text" in passages_dict:
        passages_dict["passages"] = passages_dict.pop("long_text")
    return Prediction(**passages_dict)


class Retrieve(Parameter):
    """Retrieve passages with the retriever configured in ``dspy.settings.rm``.

    ``dspy.Retrieve`` is the standard retrieval primitive used inside DSPy modules.
    It delegates the actual search to the retriever model stored in
    ``dspy.settings.rm`` and normalizes the returned passages into a
    ``dspy.Prediction`` with a ``passages`` field.

    Args:
        k: Default number of passages to request on each call.
        callbacks: Optional callbacks applied through DSPy's callback system.

    Examples:
        ```python
        import dspy

        retrieve = dspy.Retrieve(k=2)
        result = retrieve("Who wrote Pride and Prejudice?")
        print(result.passages)
        ```
    """
    name = "Search"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant passages from a corpus"

    def __init__(self, k=3, callbacks=None):
        """Initialize a retrieval parameter.

        Args:
            k: Default number of passages to retrieve.
            callbacks: Optional callback handlers for retrieval calls.
        """
        self.stage = random.randbytes(8).hex()
        self.k = k
        self.callbacks = callbacks or []

    def reset(self):
        """Reset transient retriever state.

        The base retriever does not keep mutable execution state, so this method is a
        no-op. Subclasses can override it when they need custom reset behavior.
        """
        pass

    def dump_state(self):
        """Serialize retriever configuration needed to restore this instance.

        Returns:
            dict[str, int]: The persisted retriever state.
        """
        state_keys = ["k"]
        return {k: getattr(self, k) for k in state_keys}

    def load_state(self, state):
        """Restore retriever configuration from serialized state.

        Args:
            state: State previously produced by :meth:`dump_state`.
        """
        for name, value in state.items():
            setattr(self, name, value)

    @with_callbacks
    def __call__(self, *args, **kwargs):
        """Run the retriever and return normalized passages.

        Args:
            *args: Positional arguments forwarded to :meth:`forward`.
            **kwargs: Keyword arguments forwarded to :meth:`forward`.

        Returns:
            list[str] | Prediction | list[Prediction]: The retrieval result produced by
                :meth:`forward`.
        """
        return self.forward(*args, **kwargs)

    def forward(
        self,
        query: str,
        k: int | None = None,
        **kwargs,
    ) -> list[str] | Prediction | list[Prediction]:
        """Retrieve passages for a query using the configured retriever model.

        Args:
            query: Search query passed to ``dspy.settings.rm``.
            k: Optional override for the number of passages to retrieve.
            **kwargs: Extra keyword arguments forwarded to the configured retriever.

        Returns:
            Prediction: A prediction whose ``passages`` field contains the retrieved
                passage texts.

        Raises:
            AssertionError: If no retriever model has been configured in
                ``dspy.settings.rm``.
        """
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
