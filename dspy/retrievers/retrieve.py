"""Base retrieval module for DSPy.

This module provides the base `Retrieve` class that serves as the foundation for
all retrieval operations in DSPy. It defines the interface for retrieving relevant
passages from a corpus based on a query.

Example:
    Basic usage with a configured retrieval model:

    ```python
    import dspy

    # Configure DSPy with a retrieval model
    dspy.configure(rm=your_retrieval_model)

    # Create a retriever and fetch passages
    retrieve = dspy.Retrieve(k=3)
    result = retrieve("What is machine learning?")
    print(result.passages)
    ```
"""

import random
from typing import Any

from dspy.predict.parameter import Parameter
from dspy.primitives.prediction import Prediction
from dspy.utils.callback import with_callbacks


def single_query_passage(passages: list[dict[str, Any]]) -> Prediction:
    passages_dict = {key: [] for key in list(passages[0].keys())}
    for docs in passages:
        for key, value in docs.items():
            passages_dict[key].append(value)
    if "long_text" in passages_dict:
        passages_dict["passages"] = passages_dict.pop("long_text")
    return Prediction(**passages_dict)


class Retrieve(Parameter):
    """A retrieval module that returns top passages for a given query.

    The `Retrieve` class is the base retrieval module in DSPy. It interfaces with
    a configured retrieval model (RM) to fetch relevant passages from a corpus
    based on a search query.

    Attributes:
        name: The name identifier for this module, defaults to "Search".
        input_variable: The name of the input variable, defaults to "query".
        desc: A description of what this module does.
        k: The number of top passages to retrieve.
        callbacks: A list of callback functions to run before and after retrieval.
        stage: A unique identifier for this retrieval instance.

    Example:
        Basic retrieval usage:

        ```python
        import dspy

        # Configure DSPy with a retrieval model
        dspy.configure(rm=your_retrieval_model)

        # Create a retriever
        retrieve = dspy.Retrieve(k=5)

        # Retrieve passages for a query
        result = retrieve("What are the benefits of exercise?")
        print(result.passages)  # List of top 5 relevant passages
        ```

        Using Retrieve in a DSPy module:

        ```python
        class RAG(dspy.Module):
            def __init__(self, num_passages=3):
                self.retrieve = dspy.Retrieve(k=num_passages)
                self.generate = dspy.ChainOfThought("context, question -> answer")

            def forward(self, question):
                context = self.retrieve(question).passages
                return self.generate(context=context, question=question)
        ```
    """

    name = "Search"
    input_variable = "query"
    desc = "takes a search query and returns one or more potentially relevant passages from a corpus"

    def __init__(self, k: int = 3, callbacks: list[Any] | None = None):
        self.stage = random.randbytes(8).hex()
        self.k = k
        self.callbacks = callbacks or []

    def reset(self) -> None:
        """Reset the retriever state.

        This method is a placeholder for subclasses to implement custom reset
        logic. The base implementation does nothing.
        """
        pass

    def dump_state(self) -> dict[str, Any]:
        state_keys = ["k"]
        return {k: getattr(self, k) for k in state_keys}

    def load_state(self, state: dict[str, Any]) -> None:
        """Load the retriever's state from a dictionary.

        Args:
            state: A dictionary containing the retriever's configuration state,
                typically obtained from `dump_state()`.

        Example:
            ```python
            retrieve = dspy.Retrieve(k=3)
            retrieve.load_state({"k": 10})
            # retrieve.k is now 10
            ```
        """
        for name, value in state.items():
            setattr(self, name, value)

    @with_callbacks
    def __call__(self, *args, **kwargs) -> list[str] | Prediction | list[Prediction]:
        """Execute the retrieval operation.

        This method wraps the `forward` method with callback support, allowing
        pre- and post-retrieval hooks to be executed.

        Args:
            *args: Positional arguments passed to `forward()`.
            **kwargs: Keyword arguments passed to `forward()`.

        Returns:
            The result from `forward()`, typically a Prediction object containing
            the retrieved passages.

        Example:
            ```python
            retrieve = dspy.Retrieve(k=3)
            result = retrieve("What is deep learning?")
            print(result.passages)
            ```
        """
        return self.forward(*args, **kwargs)

    def forward(
        self,
        query: str,
        k: int | None = None,
        **kwargs,
    ) -> list[str] | Prediction | list[Prediction]:
        """Retrieve top-k passages for the given query.

        This method performs the actual retrieval operation using the configured
        retrieval model (RM) in DSPy settings.

        Args:
            query: The search query string to retrieve passages for.
            k: The number of passages to retrieve. If None, uses the instance's
                default k value.
            **kwargs: Additional keyword arguments passed to the retrieval model.

        Returns:
            A Prediction object containing the retrieved passages in the
            `passages` attribute.

        Raises:
            AssertionError: If no retrieval model (RM) is configured in DSPy settings.

        Example:
            ```python
            import dspy

            # Configure DSPy with a retrieval model
            dspy.configure(rm=your_retrieval_model)

            retrieve = dspy.Retrieve(k=3)

            # Use default k
            result = retrieve.forward("What is machine learning?")
            print(result.passages)  # List of 3 passages

            # Override k for this call
            result = retrieve.forward("What is deep learning?", k=5)
            print(result.passages)  # List of 5 passages
            ```
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
