import json
from typing import Optional, Union

import dspy
from dsp.utils import dotdict

try:
    from snowflake.core import Root
except ImportError:
    raise ImportError(
        "The snowflake-snowpark-python library is required to use SnowflakeRM. Install it with dspy-ai[snowflake]",
    )


class SnowflakeRM(dspy.Retrieve):
    """A retrieval module that uses Snowlfake's Cortex Search service to return the top relevant passages for a given query.

    Assumes that a Snowflake Cortex Search endpoint has been configured by the use.

    For more information on configuring the Cortex Search service, visit: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview

    Args:
        snowflake_sesssion (object): Snowflake Snowpark session for accessing the service.
        cortex_search_service(str): Name of the Cortex Search service to be used.
        snowflake_database (str): The name of the Snowflake table containing document embeddings.
        snowflake_schema (str): The name of the Snowflake table containing document embeddings.
        search_columns (list): A comma-separated list of columns to return for each relevant result in the response. These columns must be included in the source query for the service.
        search_filter (dict): A filter object for filtering results based on data in the ATTRIBUTES columns. See Filter syntax.
        k (int, optional): The default number of top passages to retrieve. Defaults to 3.
    """

    def __init__(
        self,
        snowflake_session: object,
        cortex_search_service: str,
        snowflake_database: str,
        snowflake_schema: str,
        k: int = 3,
    ):
        super().__init__(k=k)
        self.k = k
        self.cortex_search_service_name = cortex_search_service
        self.client = self._fetch_cortex_service(
            snowflake_session, snowflake_database, snowflake_schema, cortex_search_service
        )

    def forward(
        self,
        query_or_queries: Union[str, list[str]],
        retrieval_columns: list[str],
        filter: Optional[dict] = None,
        k: Optional[int] = None,
    ) -> dspy.Prediction:
        """Query Cortex Search endpoint for top k relevant passages.
        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            retrieval_columns (List[str]): Columns to include in response.
            filter (Optional[json]):Filter query.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.
        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        k = self.k if k is None else k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]
        passages = []

        for cortex_query in queries:
            response_chunks = self._query_cortex_search(
                cortex_search_service=self.client,
                query=cortex_query,
                columns=retrieval_columns,
                filter=filter,
                k=k,
            )

            if len(retrieval_columns) == 1:
                passages.extend(
                    dotdict({"long_text": passage[self.retrieval_columns[0]]}) for passage in response_chunks["results"]
                )
            else:
                passages.extend(dotdict({"long_text": str(passage)}) for passage in response_chunks["results"])

        return passages

    def _fetch_cortex_service(self, snowpark_session, snowflake_database, snowflake_schema, cortex_search_service_name):
        """Fetch the Cortex Search service to be used"""
        snowpark_session.query_tag = {"origin": "sf_sit", "name": "dspy", "version": {"major": 1, "minor": 0}}
        root = Root(snowpark_session)

        # fetch service
        search_service = (
            root.databases[snowflake_database]
            .schemas[snowflake_schema]
            .cortex_search_services[cortex_search_service_name]
        )

        return search_service

    def _query_cortex_search(self, cortex_search_service, query, columns, filter, k):
        """Query Cortex Search endpoint for top relevant passages .

        Args:
            cortex_search_service (object): cortex search service for querying
            query (str): The query or queries to search for.
            columns (Optional[list]): A comma-separated list of columns to return for each relevant result in the response. These columns must be included in the source query for the service.
            filter (Optional[json]): A filter object for filtering results based on data in the ATTRIBUTES columns. See Filter syntax.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        # query service
        resp = cortex_search_service.search(query=query, columns=columns, filter=filter, limit=k)

        return json.loads(resp.to_json())
