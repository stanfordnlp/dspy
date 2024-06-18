"""
Retriever model for deeplake
"""

from collections import defaultdict
from typing import List, Optional, Union

import openai

import dspy
from dsp.utils import dotdict

try:
    import openai

    ERRORS = (
        openai.RateLimitError,
        openai.APIError,
    )
except Exception:
    ERRORS = (openai.RateLimitError, openai.APIError)


class DeeplakeRM(dspy.Retrieve):
    
    """
    A retriever module that uses deeplake to return the top passages for a given query.

    Assumes that a Deep Lake Vector Store has been created and populated with the following payload:
        - text: The text of the passage

    Args:
        deeplake_vectorstore_name (str): The name or path of the Deep Lake Vector Store.
        deeplake_client (VectorStore): An instance of the Deep Lake client.
        k (int, optional): The default number of top passages to retrieve. Defaults to 3.

    Examples:
        Below is a code snippet that shows how to use Deep Lake as the default retriver:
        ```python
        from deeplake import VectorStore
        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        deeplake_client = VectorStore
        retriever_model = DeeplakeRM("my_vectorstore_path", deeplake_client=deeplake_client)
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to use Deep Lake in the forward() function of a module
        ```python
        self.retrieve = DeeplakeRM("my_vectorstore_path", deeplake_client=deeplake_client, k=num_passages)
        ```
    """

    def __init__(
        self,
        deeplake_vectorstore_name: str,
        deeplake_client,
        k: int = 3,
    ):
        try:
            from deeplake import VectorStore
        except ImportError:
            raise ImportError("The 'deeplake' extra is required to use DeepLakeRM. Install it with `pip install dspy-ai[deeplake]`",)

        self._deeplake_vectorstore_name = deeplake_vectorstore_name
        self._deeplake_client = deeplake_client(
            path=self._deeplake_vectorstore_name,
            embedding_function=self.embedding_function,
            )

        super().__init__(k=k)

    def embedding_function(self, texts, model="text-embedding-ada-002"):
        if isinstance(texts, str):
            texts = [texts]

        texts = [t.replace("\n", " ") for t in texts]
        
        return [data.embedding for data in openai.embeddings.create(input = texts, model=model).data]
    
    def forward(
        self, query_or_queries: Union[str, List[str]], k: Optional[int],**kwargs,
    ) -> dspy.Prediction:
        
        """Search with DeepLake for self.k top passages for query

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [q for q in queries if q]  # Filter empty queries

        k = k if k is not None else self.k

        passages = defaultdict(float)
        #deeplake doesn't support batch querying, manually querying each query and storing them
        for query in queries:
            results = self._deeplake_client.search(query, k=k, **kwargs)

            for score,text in zip(results.get('score',0.0),results.get('text',"")):
                passages[text] += score

        sorted_passages = sorted(
            passages.items(), key=lambda x: x[1], reverse=True)[:k]
        
        return [dotdict({"long_text": p}) for p, _ in sorted_passages]