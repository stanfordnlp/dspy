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
    """
    A retrieval module that utilizes the globally configured Retriever Model (RM)
    to return passages relevant to a given search query. This class acts as a convenient
    interface for performing retrieval operations using whatever RM (retriever module)
    is set in the global `dspy.settings.rm`.

    Typical usage involves configuring a retriever (such as dspy.ColBERTv2) via `dspy.configure`,
    then creating and calling an instance of `Retrieve`. Results are returned as a
    `dspy.Prediction` object, whose `.passages` attribute is a list of retrieved documents.

    Example:
        ```python
        import dspy

        # Example: configure a retriever (e.g., ColBERTv2).
        # Note: You need to provide a valid URL or index path for your specific RM.
        colbert_rm = dspy.ColBERTv2(url='http://localhost:8893')
        dspy.configure(rm=colbert_rm)

        # Instantiate the Retrieve module for top-3 retrieval
        retrieve = dspy.Retrieve(k=3)

        # Perform retrieval
        result = retrieve("What is the capital of France?")

        # Access retrieved passages
        print(result.passages)
        # Output: ['Paris is the capital of France...', ...]
        ```

    Attributes:
        k (int): The default number of top passages to retrieve for each query.
        callbacks (list): List of callback functions to integrate with retrieval workflow.
        stage (str): Random identifier for the retrieval instance.
    """

    name = "Search"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant passages from a corpus"

    def __init__(self, k: int = 3, callbacks: list | None = None) -> None:
        """
        Initialize the Retrieve module.

        Args:
            k (int, optional): The default number of passages to retrieve for each query. Defaults to 3.
            callbacks (list, optional): Optional list of callback functions to use during retrieval.
        """
        self.stage = random.randbytes(8).hex()
        self.k = k
        self.callbacks = callbacks or []

    def reset(self) -> None:
        """Reset the retrieve module state (noop for this implementation)."""
        pass

    def dump_state(self) -> dict:
        """
        Export retriever state for checkpointing or reproducibility.

        Returns:
            dict: State dictionary including the current value of k.
        """
        state_keys = ["k"]
        return {k: getattr(self, k) for k in state_keys}

    def load_state(self, state: dict) -> None:
        """
        Load retriever state from a serialized state dictionary.

        Args:
            state (dict): Dictionary containing retriever parameters.
        """
        for name, value in state.items():
            setattr(self, name, value)

    @with_callbacks
    def __call__(self, *args, **kwargs) -> "Prediction":
        """
        Perform a retrieval using the configured RM and return a Prediction.

        Returns:
            dspy.Prediction: The prediction result containing the passages attribute (List[str]).
        """
        return self.forward(*args, **kwargs)

    def forward(
        self,
        query: str,
        k: int | None = None,
        **kwargs,
    ) -> "Prediction":
        """
        Retrieve relevant passages for a given query using the globally configured RM.

        Args:
            query (str): The search query to retrieve relevant passages for.
            k (int, optional): Overrides the default number of passages to return. If None, uses self.k.
            **kwargs: Additional keyword arguments forwarded to the RM retriever.

        Returns:
            dspy.Prediction: Prediction object with a passages attribute (List[str]) containing retrieved texts.

        Raises:
            AssertionError: If no retriever model (RM) is configured in dspy.settings.rm.
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
