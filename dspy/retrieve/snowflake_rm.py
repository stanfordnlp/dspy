from typing import Optional, Union

import dspy
from dsp.utils import dotdict

try:
    from snowflake.snowpark import Session
    from snowflake.snowpark import functions as snow_fn
    from snowflake.snowpark.functions import col, function, lit
    from snowflake.snowpark.types import VectorType

except ImportError:
    raise ImportError(
        "The snowflake-snowpark-python library is required to use SnowflakeRM. Install it with dspy-ai[snowflake]",
    )


class SnowflakeRM(dspy.Retrieve):
    """A retrieval module that uses Weaviate to return the top passages for a given query.

    Assumes that a Snowflake table has been created and populated with the following payload:
        - content: The text of the passage

    Args:
        snowflake_credentials: connection parameters for initializing Snowflake client.
        snowflake_table_name (str): The name of the Snowflake table containing document embeddings.
        embeddings_field (str): The field in the Snowflake table with the content embeddings
        embeddings_text_field (str): The field in the Snowflake table with the content.
        k (int, optional): The default number of top passages to retrieve. Defaults to 3.
    """

    def __init__(
        self,
        snowflake_table_name: str,
        snowflake_credentials: dict,
        k: int = 3,
        embeddings_field: str = "chunk_vec",
        embeddings_text_field: str = "chunk",
        embeddings_model: str = "e5-base-v2",
    ):
        self.snowflake_table_name = snowflake_table_name
        self.embeddings_field = embeddings_field
        self.embeddings_text_field = embeddings_text_field
        self.embeddings_model = embeddings_model
        self.client = self._init_cortex(credentials=snowflake_credentials)

        super().__init__(k=k)

    def forward(self, query_or_queries: Union[str, list[str]], k: Optional[int] = None) -> dspy.Prediction:
        """Search Snowflake document embeddings table for self.k top passages for query.

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]
        passages = []

        for query in queries:
            query_embeddings = self._get_embeddings(query)
            top_k_chunks = self._top_k_similar_chunks(query_embeddings, k)

            passages.extend(dotdict({"long_text": passage[0]}) for passage in top_k_chunks)

        return passages

    def _top_k_similar_chunks(self, query_embeddings, k):
        """Search Snowflake table for self.k top passages for query.

        Args:
            query_embeddings(List[float]]): the embeddings for the query of interest
            doc_table
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        doc_table_value = self.embeddings_field
        doc_table_key = self.embeddings_text_field

        doc_embeddings = self.client.table(self.snowflake_table_name)
        cosine_similarity = function("vector_cosine_similarity")

        top_k = (
            doc_embeddings.select(
                doc_table_value,
                doc_table_key,
                cosine_similarity(
                    doc_embeddings.col(doc_table_value),
                    lit(query_embeddings).cast(VectorType(float, len(query_embeddings))),
                ).as_("dist"),
            )
            .sort("dist", ascending=False)
            .limit(k)
        )

        return top_k.select(doc_table_key).to_pandas().values

    @classmethod
    def _init_cortex(cls, credentials: dict) -> None:
        session = Session.builder.configs(credentials).create()
        session.query_tag = {"origin": "sf_sit", "name": "dspy", "version": {"major": 1, "minor": 0}}

        return session

    def _get_embeddings(self, query: str) -> list[float]:
        # create embeddings for the query
        embed = snow_fn.builtin("snowflake.cortex.embed_text_768")
        cortex_embed_args = embed(snow_fn.lit(self.embeddings_model), snow_fn.lit(query))

        return self.client.range(1).withColumn("complete_cal", cortex_embed_args).collect()[0].COMPLETE_CAL
