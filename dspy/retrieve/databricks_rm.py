import json
import os
from typing import List, Union, Any, Dict
import requests
import dspy
from dspy.primitives.prediction import Prediction


class DatabricksRM(dspy.Retrieve):
    """
    A retrieval module that uses Databricks Vector Search Endpoint to return the top-k embeddings for a given query.

    Args:
        databricks_index_name (str): Databricks vector search index to query
        databricks_endpoint (str): Databricks index endpoint url
        databricks_token (str): Databricks authentication token
        columns (list[str]): Column names to include in response
        filters_json (str, optional): JSON string for query filters
        k (int, optional): Number of top embeddings to retrieve. Defaults to 3.
        docs_id_column_name (str, optional): Column name for retrieved doc_ids to return.
        text_column_name (str, optional): Column name for retrieved text to return.

    Examples:
        Below is a code snippet that shows how to configure Databricks Vector Search endpoints:

        (example adapted from "Databricks: How to create and query a Vector Search Index:
        https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#create-a-vector-search-index)

        ```python
        from databricks.vector_search.client import VectorSearchClient

        #Creating Vector Search Client

        client = VectorSearchClient()

        client.create_endpoint(
            name="your_vector_search_endpoint_name",
            endpoint_type="STANDARD"
        )

        #Creating Vector Search Index using Python SDK
        #Example for Direct Vector Access Index

        index = client.create_direct_access_index(
            endpoint_name="your_databricks_host_url",
            index_name="your_index_name",
            primary_key="id",
            embedding_dimension=1024,
            embedding_vector_column="text_vector",
            schema={
            "id": "int",
            "field2": "str",
            "field3": "float",
            "text_vector": "array<float>"}
        )

        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        retriever_model = DatabricksRM(databricks_index_name = "your_index_name",
        databricks_endpoint = "your_databricks_host_url", databricks_token = "your_databricks_token", columns= ["id", "field2", "field3", "text_vector"], k=3)
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to query the Databricks Direct Vector Access Index using the forward() function.
        ```python
        self.retrieve = DatabricksRM(query=[1, 2, 3], query_type = 'vector')
        ```
    """

    def __init__(
        self,
        databricks_index_name=None,
        databricks_endpoint=None,
        databricks_token=None,
        columns=None,
        filters_json=None,
        k=3,
        docs_id_column_name="id",
        text_column_name="text",
    ):
        super().__init__(k=k)
        if not databricks_token and not os.environ.get("DATABRICKS_TOKEN"):
            raise ValueError(
                "You must supply databricks_token or set environment variable DATABRICKS_TOKEN"
            )
        if not databricks_endpoint and not os.environ.get("DATABRICKS_HOST"):
            raise ValueError(
                "You must supply databricks_endpoint or set environment variable DATABRICKS_HOST"
            )
        if not databricks_index_name:
            raise ValueError("You must supply vector index name")
        if not columns:
            raise ValueError(
                "You must specify a list of column names to be included in the response"
            )
        self.databricks_token = (
            databricks_token if databricks_token else os.environ["DATABRICKS_TOKEN"]
        )
        self.databricks_endpoint = (
            databricks_endpoint
            if databricks_endpoint
            else os.environ["DATABRICKS_HOST"]
        )
        self.databricks_index_name = databricks_index_name
        self.columns = columns
        self.filters_json = filters_json
        self.k = k
        self.docs_id_column_name = docs_id_column_name
        self.text_column_name = text_column_name

    def _extract_doc_ids(self, item: Dict[str, Any]) -> str:
        """Extracts the document id from a search result

        Args:
            item: Dict[str, Any]: a record from the search results.
        Returns:
            str: document id.
        """
        if self.docs_id_column_name == "metadata":
            docs_dict = json.loads(item["metadata"])
            return docs_dict["document_id"]
        return item[self.docs_id_column_name]

    def _get_extra_columns(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts search result column values, excluding the "text" and not "id" columns

        Args:
            item: Dict[str, Any]: a record from the search results.
        Returns:
            Dict[str, Any]: Search result column values, excluding the "text" and not "id" columns.
        """
        extra_columns = {
            k: v
            for k, v in item.items()
            if k not in [self.docs_id_column_name, self.text_column_name]
        }
        if self.docs_id_column_name == "metadata":
            extra_columns = {
                **extra_columns,
                **{
                    "metadata": {
                        k: v
                        for k, v in json.loads(item["metadata"]).items()
                        if k != "document_id"
                    }
                },
            }
        return extra_columns

    def forward(
        self,
        query: Union[str, List[float]],
        query_type: str = "text",
        filters_json: str = None,
    ) -> dspy.Prediction:
        """Search with Databricks Vector Search Client for self.k top results for query

        Args:
            query (Union[str, List[float]]): query to search for.
            query_type (str): 'vector' for Direct Vector Access Index and Delta Sync Index using self-managed vectors or 'text' for Delta Sync Index using model endpoint.

        Returns:
            dspy.Prediction: An object containing the retrieved results.
        """
        headers = {
            "Authorization": f"Bearer {self.databricks_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "columns": self.columns,
            "num_results": self.k,
        }
        if query_type == "vector":
            if not isinstance(query, list):
                raise ValueError("Query must be a list of floats for query_vector")
            payload["query_vector"] = query
        elif query_type == "text":
            if not isinstance(query, str):
                raise ValueError("Query must be a string for query_text")
            payload["query_text"] = query
        else:
            raise ValueError("Invalid query type specified. Use 'vector' or 'text'.")

        payload["filters_json"] = filters_json if filters_json else self.filters_json

        response = requests.post(
            f"{self.databricks_endpoint}/api/2.0/vector-search/indexes/{self.databricks_index_name}/query",
            json=payload,
            headers=headers,
        )
        results = response.json()

        # Check for errors from REST API call
        if response.json().get("error_code", None) != None:
            raise Exception(
                f"ERROR: {response.json()['error_code']} -- {response.json()['message']}"
            )

        # Checking if defined columns are present in the index columns
        col_names = [column["name"] for column in results["manifest"]["columns"]]

        if self.docs_id_column_name not in col_names:
            raise Exception(
                f"docs_id_column_name: '{self.docs_id_column_name}' is not in the index columns: \n {col_names}"
            )

        if self.text_column_name not in col_names:
            raise Exception(
                f"text_column_name: '{self.text_column_name}' is not in the index columns: \n {col_names}"
            )

        # Extracting the results
        items = []
        for idx, data_row in enumerate(results["result"]["data_array"]):
            item = {}
            for col_name, val in zip(col_names, data_row):
                item[col_name] = val
            items += [item]

        # Sorting results by score in descending order
        sorted_docs = sorted(items, key=lambda x: x["score"], reverse=True)[:self.k]

        # Returning the prediction
        return Prediction(
            docs=[doc[self.text_column_name] for doc in sorted_docs],
            doc_ids=[
                self._extract_doc_ids(doc)
                for doc in sorted_docs
            ],
            extra_columns=[self._get_extra_columns(item) for item in sorted_docs],
        )
