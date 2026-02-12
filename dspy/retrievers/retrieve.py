import random

from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction
from dspy.utils.callback import with_callbacks


def single_query_passage(passages):
    """Merge a list of passage dicts into a single Prediction with aggregated fields.

    Takes a list of passage dictionaries (one per retrieved document) and merges
    them into a single Prediction where each key maps to a list of values. The
    ``long_text`` key is renamed to ``passages`` for consistency.

    Args:
        passages: A list of dictionaries, each representing a retrieved passage.
            All dictionaries should share the same keys.

    Returns:
        A ``Prediction`` with aggregated fields. For example, if each passage dict
        has keys ``long_text`` and ``score``, the result will have ``passages``
        (list of texts) and ``score`` (list of scores).

    Example:
        >>> docs = [{"long_text": "Doc 1", "score": 0.9}, {"long_text": "Doc 2", "score": 0.7}]
        >>> result = single_query_passage(docs)
        >>> result.passages
        ['Doc 1', 'Doc 2']
    """
    passages_dict = {key: [] for key in list(passages[0].keys())}
    for docs in passages:
        for key, value in docs.items():
            passages_dict[key].append(value)
    if "long_text" in passages_dict:
        passages_dict["passages"] = passages_dict.pop("long_text")
    return Prediction(**passages_dict)


class Retrieve(Parameter):
    """Retrieval module that searches a corpus for passages relevant to a query.

    ``Retrieve`` is the standard retrieval interface in DSPy. It delegates the
    actual search to the retrieval model (RM) configured via ``dspy.settings``,
    making it agnostic to the underlying retrieval backend.

    Args:
        k: The number of top passages to retrieve. Defaults to 3.
        callbacks: Optional list of callback functions invoked during retrieval.

    Example:
        >>> import dspy
        >>> dspy.settings.configure(rm=my_retriever)
        >>> retrieve = Retrieve(k=5)
        >>> result = retrieve("What is the capital of France?")
        >>> result.passages
        ['Paris is the capital...', ...]
    """

    name = "Search"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant passages from a corpus"

    def __init__(self, k=3, callbacks=None):
        """Initialize the Retrieve module.

        Args:
            k: Number of top passages to retrieve. Defaults to 3.
            callbacks: Optional list of callback functions.
        """
        self.stage = random.randbytes(8).hex()
        self.k = k
        self.callbacks = callbacks or []

    def reset(self):
        """Reset the module state. Currently a no-op."""
        pass

    def dump_state(self):
        """Serialize the module's configurable state to a dictionary.

        Returns:
            A dictionary containing the module's configuration (e.g., ``{"k": 3}``).
        """
        state_keys = ["k"]
        return {k: getattr(self, k) for k in state_keys}

    def load_state(self, state):
        """Restore the module's state from a dictionary.

        Args:
            state: A dictionary of attribute names to values, as returned
                by ``dump_state()``.
        """
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
        """Execute a retrieval query against the configured retrieval model.

        Delegates to the RM set in ``dspy.settings.rm`` and returns the top-k
        passages as a ``Prediction``.

        Args:
            query: The search query string.
            k: Number of passages to retrieve. If ``None``, uses the instance
                default (``self.k``).
            **kwargs: Additional keyword arguments forwarded to the retrieval model.

        Returns:
            A ``Prediction`` with a ``passages`` field containing the retrieved
            text passages.

        Raises:
            AssertionError: If no retrieval model is configured in ``dspy.settings``.
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
