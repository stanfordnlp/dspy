from collections import defaultdict
from typing import List, Union
import dspy
from typing import Optional

# Note, using Python v3 Weaviate Client -- need to make sure this is tied to `pip install dspy-ai[weaviate]`

try:
    import weaviate
except ImportError:
    raise ImportError(
        "The 'weaviate' extra is required to use WeaviateRM. Install it with `pip install dspy-ai[weaviate]`"
    )


class WeaviateRM(dspy.Retrieve):
    """
    A retrieval module that uses Weaviate to return the top passages for a given query.

    Assumes that a Weaviate collection has been created and populated with the following payload:
        - content: The text of the passage

    Args:
        weaviate_collection_name (str): The name of the Weaviate collection.
        weaviate_client (WeaviateClient): An instance of the Weaviate client.
        k (int, optional): The default number of top passages to retrieve. Defaults to 3.

    Examples:
        Below is a code snippet that shows how to use Weaviate as the default retriver:
        ```python
        import weaviate

        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        weaviate_client = weaviate.Client("your-path-here")
        retriever_model = WeaviateRM("my_collection_name", weaviate_client=weaviate_client)
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to use Weaviate in the forward() function of a module
        ```python
        self.retrieve = WeaviateRM("my_collection_name", weaviate_client=weaviate_client, k=num_passages)
        ```
    """

    def __init__(self, 
                 weaviate_collection_name: str, 
                 weaviate_client: weaviate.Client, 
                 k: int = 3
        ):
        self._weaviate_collection_name = weaviate_collection_name
        self._weaviate_client = weaviate_client

        super().__init__(k=k)

    # ToDo -- interface query batching
    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int]) -> dspy.Prediction:
        """Search with Weaviate for self.k top passages for query

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.
        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """

        # Please note, as stated above the WeaviateRM assumes you have a text key payload named: `content`
        k = k if k is not None else self.k
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [q for q in queries if q]  # Filter empty queries
        passages = []
        # Note this assumes you are using a text2vec embedding service orchestrated in Weaviate
        for query in queries:
            # We are currently hard-coding the payload / text key `content`
            results = self._weaviate_client.query\
                .get(self._weaviate_collection_name, ["content"])\
                .with_near_text({"concepts": [query]})\
                .with_limit(k)\
                .do()

            results = results["data"]["Get"][self._weaviate_collection_name]
            parsed_results = [result["content"] for result in results]
            # This assumes you will call `dspy.utils.deduplicate` in the forward pass
            # I am skeptical how often you will have duplicate documents,
            #   but an idea could be to extend this with the distance scores.
            #   However, it probably makes more sense for us (Weaviate) to cook this into our
            #   batch query API (which will be in the v4 client, expected to go GA in 1.24).
            passages.extend(parsed_results)

        return dspy.Prediction(passages=passages)
