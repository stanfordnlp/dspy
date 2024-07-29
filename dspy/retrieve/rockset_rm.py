from typing import List, Optional, Union

import dspy
import pandas as pd
from dsp.utils import dotdict

try:
    import rockset
    from rockset import ApiException, Regions, RocksetClient
except ImportError:
    raise ImportError(
        "The 'rockset' extra is required to use RocksetRM. Install it with `pip install dspy-ai[rockset]`",
    )


class RocksetRM(dspy.Retrieve):
    """
    A retrieval module that uses Rockset to return the top passages for a given query.

    Assumes that a Rockset collection has been created and populated with the following payload:
        - content: The text of the passage

    Args:
        rockset_collection_name (str): The name of the Rockset collection.
        rockset_client (RocksetClient): An instance of the Rockset client.
        k (int, optional): The default number of top passages to retrieve. Defaults to 3.
        rockset_collection_text_key (str, optional): The key in the collection with the content. Defaults to
        page_content.
        rockset_collection_source_key (str, optional): The key in the collection with the source. Defaults to
        source.
        rockset_collection_embeddings_key (str, optional): The key in the collection with the embeddings. Defaults to
        embeddings.

    Examples:
        Below is a code snippet that shows how to use Rockset as the default retriver:
        ```python
        import rockset

        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        rockset_client = rockset.Client("your-path-here")
        retriever_model = RocksetRM(rockset_collection_name="my_collection_name",
                                     rockset_collection_text_key="content",
                                     rockset_client=rockset_client)
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to use Rockset in the forward() function of a module
        ```python
        self.retrieve = RocksetRM("my_collection_name", rockset_client=rockset_client, k=num_passages)
        ```
    """

    def __init__(self,
                 rockset_collection_name: str,
                 rockset_client: rockset.RocksetClient,
                 k: int = 3,
                 rockset_collection_text_key: Optional[str] = "page_content",
                 rockset_collection_source_key: Optional[str] = "source",
                 rockset_collection_embeddings_key: Optional[str] = "embeddings",
        ):
        self._rockset_collection_name = rockset_collection_name
        self._rockset_client = rockset_client
        self._rockset_collection_text_key = rockset_collection_text_key
        self._rockset_collection_source_key = rockset_collection_source_key
        self._rockset_collection_embeddings_key = rockset_collection_embeddings_key
        super().__init__(k=k)

    def forward(self, query_or_queries: Union[str, List[str]], metadata_filter: Optional[str] = None,
                k: Optional[int] = None, **kwargs) -> dspy.Prediction:
        """Search with Rockset for self.k top passages for query

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            metadata_filter (Optional[str]): a filter on the metadata within the vectorstore.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.
        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """

        k = k if k is not None else self.k
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [q for q in queries if q]
        passages = []

        for query in queries:
            sim_query = f"""
                    with base as
                    (select {self._rockset_collection_text_key}, {self._rockset_collection_source_key}, 
                    COSINE_SIM(v.{self._rockset_collection_embedding_key}, {query.strip("'")}) as sim_search 
                    from {self._rockset_collection_name} v 
                    where v.{self._rockset_collection_embedding_key} is not null
                    {metadata_filter}
                    order by sim_search desc limit {k}) 
                    select {self._rockset_collection_text_key}, {self._rockset_collection_source_key}, 
                    sim_search from base
                    """

            results = self._rockset_client.sql(query=sim_query)

            if not pd.DataFrame(results["results"], dtype='str').empty:
                data = pd.DataFrame(results["results"], dtype='str')[[self._rockset_collection_text_key,
                                                                      self._rockset_collection_source_key]].values
                passages.extend(data)

        # Return type not changed, needs to be a Prediction object. But other code will break if we change it.
        return passages
