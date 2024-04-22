import dspy

try:
    from llama_index.core.base.base_retriever import BaseRetriever
except ImportError:
    err = "The 'llama_index' package is required to use LlamaIndexRM. Install it with 'pip install llama_index'."
    raise ImportError(err) from None


class LlamaIndexRM(dspy.Retrieve):
    """Implements a retriever which wraps over a LlamaIndex retriever.

    This is done to bridge LlamaIndex and DSPy and allow the various retrieval
    abstractions in LlamaIndex to be used in DSPy.

    To-do (maybe):
        - Async support (DSPy lacks this entirely it seems, so not a priority until the rest of the repo catches on)
        - Text/video retrieval (Available in LI, not sure if this will be a priority in DSPy)

    Args:
        retriever (BaseRetriever): A LlamaIndex retriever object - text based only

    Returns:
        DSPy RM Object - this is a retriever object that can be used in DSPy
    """

    retriever: BaseRetriever

    def __init__(self, retriever: BaseRetriever):
        self.retriever = retriever

    @property
    def similarity_top_k(self) -> int:
        """Return similarity top k of retriever."""
        return self.retriever.similarity_top_k

    @similarity_top_k.setter
    def similarity_top_k(self, k: int) -> None:
        """Set similarity top k of retriever."""
        self.retriever.similarity_top_k = k

    def forward(self, query: str) -> list[dspy.Example]:
        """Forward function for the LI retriever.

        This is the function that is called to retrieve the top k examples for a given query.

        Top k is set via the setter similarity_top_k or at LI instantiation.

        Args:
            query (str): The query to retrieve examples for

        Returns:
            List[dspy.Example]: A list of examples retrieved by the retriever
        """
        raw = self.retriever.retrieve(query)

        return [
            dspy.Example(
                text=result.text,
                score=result.score,
            )
            for result in raw
        ]
