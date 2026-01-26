"""
Simple LLM reranker module.

This is NOT recommended in production.
If possible, leverage a cross-encoder (i.e ModernBERT).

This was originally done as an example reranker for BingRM. 
I did not want to involve other libraries like sentence-transformers (although this is ideal).

Example usage:
```python
    passages = [ #automatically casts into string, presuming the class supports that
        "Machine learning algorithms use statistical methods to find patterns in large datasets.,
        "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
        "Data preprocessing is crucial for ensuring machine learning models perform accurately.",
        Foo("Natural language processing combines linguistics and machine learning techniques."),
        Foo("Reinforcement learning involves agents learning through interaction with an environment."),
    ]

    lm = dspy.LM(
        model="openai/gpt-4o-mini",
    )
    dspy.configure(lm=lm)

    reranker = LLMReranker()
    reranked_passages = reranker.forward(query, passages)
    print(reranked_passages)
```
"""

import dspy

class Similarity(dspy.Signature):
    """A simple similarity ranker which compares a document to a given query
    and returns their similarity score (1-10)."""

    document: str = dspy.InputField(
        description="The document to compare against the original query."
    )
    query: str = dspy.InputField(
        description="The original query."
    )

    similarity: int = dspy.OutputField(
        description="The similarity score between the document and the query."
    )

class LLMReranker(dspy.Module):
    """A simple reranker that uses a language model to rerank the top passages (1-10)
    retrieved by a retriever.
    
    This is NOT recommended for production use.
    """

    def __init__(self):
        self.similarity = dspy.Predict(Similarity)

    def forward(self, query: str, passages: list) -> list:
        """Rerank the top passages retrieved by a retriever.

        Args:
            query (str): The original query.
            passages (list): The top passages retrieved by a retriever.

        Returns:
            list: The reranked passages.
        """

        # Get the similarity score for each passage
        similarities = [
            min( # Ensure the similarity score is between 1-10
                self.similarity(
                    document=str(passage)
                    , query=query
                ).similarity
                , 10
            )
            for passage in passages
        ]

        # Sort the passages by similarity score
        reranked = [
            passage
            for _, passage in sorted(
                zip(similarities, passages)
                , key=lambda x: x[0]
                , reverse=True
            )
        ]

        return reranked

