import json
import os
from importlib.util import find_spec
from typing import Any, Dict, List, Optional

import requests

from dspy.retriever import Retriever
from dspy.primitives.prediction import Prediction
from dspy.clients.embedding import Embedding

_databricks_sdk_installed = find_spec("databricks.sdk") is not None


class Databricks(Retriever):
    """
    A retriever module that uses a Databricks Mosaic AI Vector Search Index to return the top-k
    embeddings for a given query.

    Examples:
        Below is a code snippet that shows how to set up a Databricks Vector Search Index
        and configure a Databricks retriever module to query the index.

        (example adapted from "Databricks: How to create and query a Vector Search Index:
        https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-index)

        ```python
        from databricks.vector_search.client import VectorSearchClient

        # Create a Databricks Vector Search Endpoint
        client = VectorSearchClient()
        client.create_endpoint(
            name="your_vector_search_endpoint_name",
            endpoint_type="STANDARD"
        )

        # Create a Databricks Direct Access Vector Search Index
        index = client.create_direct_access_index(
            endpoint_name="your_vector_search_endpoint_name",
            index_name="your_index_name",
            primary_key="id",
            embedding_dimension=1024,
            embedding_vector_column="text_vector",
            schema={
              "id": "int",
              "field2": "str",
              "field3": "float",
              "text_vector": "array<float>"
            }
        )

        # Create a Databricks retriever module to query the Databricks Direct Access Vector
        # Search Index
        from dspy.retriever.databricks import Databricks
        
        retriever = Databricks(
            databricks_index_name = "your_index_name",
            docs_id_column_name="id",
            text_column_name="field2",
            k=3
        )
        ```

        Below is a code snippet that shows how to query the Databricks Direct Access Vector
        Search Index using the Databricks retriever module:

        ```python
        retrieved_results = retriever(query="Example query text")
        ```
    """

    def __init__(
        self,
        databricks_index_name: str,
        databricks_endpoint: Optional[str] = None,
        databricks_token: Optional[str] = None,
        columns: Optional[List[str]] = None,
        filters_json: Optional[str] = None,
        query_type: str = "ANN",
        k: int = 3,
        docs_id_column_name: str = "id",
        text_column_name: str = "text",
        embedder: Optional[Embedding] = None,
        callbacks: Optional[List[Any]] = None,
        cache: bool = False,
    ):
        """
        Args:
            databricks_index_name (str): The name of the Databricks Vector Search Index to query.
            databricks_endpoint (Optional[str]): The URL of the Databricks Workspace containing
                the Vector Search Index. Defaults to the value of the ``DATABRICKS_HOST``
                environment variable. If unspecified, the Databricks SDK is used to identify the
                endpoint based on the current environment.
            databricks_token (Optional[str]): The Databricks Workspace authentication token to use
                when querying the Vector Search Index. Defaults to the value of the
                ``DATABRICKS_TOKEN`` environment variable. If unspecified, the Databricks SDK is
                used to identify the token based on the current environment.
            columns (Optional[List[str]]): Extra column names to include in response,
                in addition to the document id and text columns specified by
                ``docs_id_column_name`` and ``text_column_name``.
            filters_json (Optional[str]): A JSON string specifying additional query filters.
                Example filters: ``{"id <": 5}`` selects records that have an ``id`` column value
                less than 5, and ``{"id >=": 5, "id <": 10}`` selects records that have an ``id``
                column value greater than or equal to 5 and less than 10.
            query_type (str): The type of search query to perform. Must be 'ANN', 'HYBRID', or 'VECTOR'.
            k (int): The number of documents to retrieve. Defaults to 3
            docs_id_column_name (str): The name of the column in the Databricks Vector Search Index
                containing document IDs.
            text_column_name (str): The name of the column in the Databricks Vector Search Index
                containing document text to retrieve.
            embedder (Optional[Embedding]): An embedder to convert query text to vectors when using 'VECTOR' query_type.
            callbacks (Optional[List[Any]]): List of callback functions.
            cache (bool, optional): Enable retrieval caching. Disabled by default.
        """
        super().__init__(embedder=embedder, k=k, callbacks=callbacks, cache=cache)
        self.databricks_token = databricks_token if databricks_token is not None else os.environ.get("DATABRICKS_TOKEN")
        self.databricks_endpoint = (
            databricks_endpoint if databricks_endpoint is not None else os.environ.get("DATABRICKS_HOST")
        )
        if not _databricks_sdk_installed and (self.databricks_token, self.databricks_endpoint).count(None) > 0:
            raise ValueError(
                "To retrieve documents with Databricks Vector Search, you must install the"
                " databricks-sdk Python library, supply the databricks_token and"
                " databricks_endpoint parameters, or set the DATABRICKS_TOKEN and DATABRICKS_HOST"
                " environment variables."
            )
        self.databricks_index_name = databricks_index_name
        self.columns = list({docs_id_column_name, text_column_name, *(columns or [])})
        self.filters_json = filters_json
        self.query_type = query_type
        self.docs_id_column_name = docs_id_column_name
        self.text_column_name = text_column_name

    def _extract_doc_ids(self, item: Dict[str, Any]) -> str:
        """Extracts the document id from a search result.

        Args:
            item (Dict[str, Any]): A record from the search results.

        Returns:
            str: Document id.
        """
        if self.docs_id_column_name == "metadata":
            docs_dict = json.loads(item["metadata"])
            return docs_dict["document_id"]
        return item[self.docs_id_column_name]

    def _get_extra_columns(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts search result column values, excluding the "text" and "id" columns.

        Args:
            item (Dict[str, Any]): A record from the search results.

        Returns:
            Dict[str, Any]: Search result column values, excluding the "text" and "id" columns.
        """
        extra_columns = {k: v for k, v in item.items() if k not in [self.docs_id_column_name, self.text_column_name]}
        if self.docs_id_column_name == "metadata":
            extra_columns = {
                **extra_columns,
                **{"metadata": {k: v for k, v in json.loads(item["metadata"]).items() if k != "document_id"}},
            }
        return extra_columns

    def forward(self, query: str, k: Optional[int] = None) -> Prediction:
        """
        Retrieve documents from a Databricks Mosaic AI Vector Search Index that are relevant to the
        specified query.

        Args:
            query (str): The query text for which to retrieve relevant documents.
            k (Optional[int]): The number of documents to retrieve. If None, defaults to self.k.

        Returns:
            dspy.Prediction: An object containing the retrieved results.
        """
        k = k or self.k
        query_text = query
        query_vector = None

        if self.query_type.upper() in ["ANN", "HYBRID"]:
            query_text = query
        elif self.query_type.upper() == "VECTOR":
            if self.embedder:
                query_vector = self.embedder.embed(query)
                query_text = None
            else:
                raise ValueError("An embedder must be provided when using 'VECTOR' query_type without providing a query vector.")
        else:
            raise ValueError(f"Unsupported query_type: {self.query_type}")

        if _databricks_sdk_installed:
            results = self._query_via_databricks_sdk(
                index_name=self.databricks_index_name,
                k=k,
                columns=self.columns,
                query_type=self.query_type.upper(),
                query_text=query_text,
                query_vector=query_vector,
                databricks_token=self.databricks_token,
                databricks_endpoint=self.databricks_endpoint,
                filters_json=self.filters_json,
            )
        else:
            results = self._query_via_requests(
                index_name=self.databricks_index_name,
                k=k,
                columns=self.columns,
                databricks_token=self.databricks_token,
                databricks_endpoint=self.databricks_endpoint,
                query_type=self.query_type.upper(),
                query_text=query_text,
                query_vector=query_vector,
                filters_json=self.filters_json,
            )

        # Checking if defined columns are present in the index columns
        col_names = [column["name"] for column in results["manifest"]["columns"]]

        if self.docs_id_column_name not in col_names:
            raise Exception(
                f"docs_id_column_name: '{self.docs_id_column_name}' is not in the index columns: \n {col_names}"
            )

        if self.text_column_name not in col_names:
            raise Exception(f"text_column_name: '{self.text_column_name}' is not in the index columns: \n {col_names}")

        # Extracting the results
        items = []
        for data_row in results["result"]["data_array"]:
            item = {col_name: val for col_name, val in zip(col_names, data_row)}
            items.append(item)

        # Sorting results by score in descending order
        sorted_docs = sorted(items, key=lambda x: x["score"], reverse=True)[:k]

        # Returning the prediction
        return Prediction(
            passages=[doc[self.text_column_name] for doc in sorted_docs],
            doc_ids=[self._extract_doc_ids(doc) for doc in sorted_docs],
            extra_columns=[self._get_extra_columns(doc) for doc in sorted_docs],
        )

    @staticmethod
    def _query_via_databricks_sdk(
        index_name: str,
        k: int,
        columns: List[str],
        query_type: str,
        query_text: Optional[str],
        query_vector: Optional[List[float]],
        databricks_token: Optional[str],
        databricks_endpoint: Optional[str],
        filters_json: Optional[str],
    ) -> Dict[str, Any]:
        """
        Query a Databricks Vector Search Index via the Databricks SDK.
        Assumes that the databricks-sdk Python library is installed.

        Args:
            index_name (str): Name of the Databricks vector search index to query
            k (int): Number of relevant documents to retrieve.
            columns (List[str]): Column names to include in response.
            query_text (Optional[str]): Text query for which to find relevant documents. Exactly
                one of query_text or query_vector must be specified.
            query_vector (Optional[List[float]]): Numeric query vector for which to find relevant
                documents. Exactly one of query_text or query_vector must be specified.
            filters_json (Optional[str]): JSON string representing additional query filters.
            databricks_token (str): Databricks authentication token. If not specified,
                the token is resolved from the current environment.
            databricks_endpoint (str): Databricks index endpoint url. If not specified,
                the endpoint is resolved from the current environment.
        Returns:
            Dict[str, Any]: Parsed JSON response from the Databricks Vector Search Index query.
        """
        from databricks.sdk import WorkspaceClient

        if (query_text, query_vector).count(None) != 1:
            raise ValueError("Exactly one of query_text or query_vector must be specified.")

        databricks_client = WorkspaceClient(host=databricks_endpoint, token=databricks_token)
        return databricks_client.vector_search_indexes.query_index(
            index_name=index_name,
            query_type=query_type,
            query_text=query_text,
            query_vector=query_vector,
            columns=columns,
            filters_json=filters_json,
            num_results=k,
        ).as_dict()

    @staticmethod
    def _query_via_requests(
        index_name: str,
        k: int,
        columns: List[str],
        databricks_token: str,
        databricks_endpoint: str,
        query_type: str,
        query_text: Optional[str],
        query_vector: Optional[List[float]],
        filters_json: Optional[str],
    ) -> Dict[str, Any]:
        """
        Query a Databricks Vector Search Index via the Python requests library.

        Args:
            index_name (str): Name of the Databricks vector search index to query
            k (int): Number of relevant documents to retrieve.
            columns (List[str]): Column names to include in response.
            databricks_token (str): Databricks authentication token.
            databricks_endpoint (str): Databricks index endpoint url.
            query_text (Optional[str]): Text query for which to find relevant documents. Exactly
                one of query_text or query_vector must be specified.
            query_vector (Optional[List[float]]): Numeric query vector for which to find relevant
                documents. Exactly one of query_text or query_vector must be specified.
            filters_json (Optional[str]): JSON string representing additional query filters.

        Returns:
            Dict[str, Any]: Parsed JSON response from the Databricks Vector Search Index query.
        """
        if (query_text, query_vector).count(None) != 1:
            raise ValueError("Exactly one of query_text or query_vector must be specified.")

        headers = {
            "Authorization": f"Bearer {databricks_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "columns": columns,
            "num_results": k,
            "query_type": query_type,
        }
        if filters_json is not None:
            payload["filters_json"] = filters_json
        if query_text is not None:
            payload["query_text"] = query_text
        elif query_vector is not None:
            payload["query_vector"] = query_vector
        response = requests.post(
            f"{databricks_endpoint}/api/2.0/vector-search/indexes/{index_name}/query",
            json=payload,
            headers=headers,
        )
        results = response.json()
        if "error_code" in results:
            raise Exception(f"ERROR: {results['error_code']} -- {results['message']}")
        return results
