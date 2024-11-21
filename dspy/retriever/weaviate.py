from typing import Any, List, Optional, Union

from dspy.retriever import Retriever
from dspy.primitives.prediction import Prediction
from dsp.utils import dotdict

try:
    import weaviate
    from weaviate.util import get_valid_uuid
    from uuid import uuid4
except ImportError as err:
    raise ImportError(
        "The 'weaviate' extra is required to use Weaviate. Install it with `pip install dspy-ai[weaviate]`",
    ) from err


class Weaviate(Retriever):
    """A retrieval module that uses Weaviate to return the top passages for a given query.

    Assumes that a Weaviate collection has been created and populated with the following payload:
        - content: passage text

    Args:
        weaviate_collection_name (str): Name of the Weaviate collection.
        weaviate_client (Union[weaviate.WeaviateClient, weaviate.Client]): An instance of the Weaviate client.
        weaviate_collection_text_key (Optional[str]): The key in the Weaviate collection where the passage text is stored. Defaults to "content".
        k (int): Number of top passages to retrieve. Defaults to 5.
        callbacks (Optional[List[Any]]): List of callback functions.
        cache (bool): Enable retrieval caching. Disabled by default.

    Examples:
        Below is a code snippet that shows how to use this retriever:
        ```python
        import weaviate
        import dspy
        from dspy.retriever.weaviate import Weaviate

        weaviate_client = weaviate.Client("http://localhost:8080")
        retriever = Weaviate(
            weaviate_collection_name="MyCollection",
            weaviate_client=weaviate_client,
            k=5
        )
        results = retriever("What are the stages in planning public works?").passages
        print(results)
        ```
    """

    def __init__(
        self,
        weaviate_collection_name: str,
        weaviate_client: Union[weaviate.WeaviateClient, weaviate.Client],
        weaviate_collection_text_key: Optional[str] = "content",
        k: int = 5,
        callbacks: Optional[List[Any]] = None,
        cache: bool = False,
    ):
        super().__init__(embedder=None, k=k, callbacks=callbacks, cache=cache)
        self._weaviate_collection_name = weaviate_collection_name
        self._weaviate_client = weaviate_client
        self._weaviate_collection_text_key = weaviate_collection_text_key

        # Determine client type (Weaviate v3 or v4)
        if hasattr(weaviate_client, "collections"):
            self._client_type = "WeaviateClient"  # Weaviate v4
        elif hasattr(weaviate_client, "query"):
            self._client_type = "Client"  # Weaviate v3
        else:
            raise ValueError("Unsupported Weaviate client type")

    def forward(self, query: Union[str, List[str]], k: Optional[int] = None) -> Prediction:
        """Search with Weaviate for the top k passages for the query or queries.

        Args:
            query (Union[str, List[str]]): The query or list of queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        k = k if k is not None else self.k
        queries = [query] if isinstance(query, str) else query
        queries = [q for q in queries if q]
        passages = []

        for q in queries:
            if self._client_type == "WeaviateClient":
                # For Weaviate v4
                results = (
                    self._weaviate_client.query.get(
                        self._weaviate_collection_name,
                        [self._weaviate_collection_text_key],
                    )
                    .with_hybrid(query=q)
                    .with_limit(k)
                    .do()
                )

                parsed_results = [
                    result[self._weaviate_collection_text_key]
                    for result in results["data"]["Get"][self._weaviate_collection_name]
                ]

            elif self._client_type == "Client":
                # For Weaviate v3
                results = (
                    self._weaviate_client.query.get(
                        self._weaviate_collection_name,
                        [self._weaviate_collection_text_key],
                    )
                    .with_hybrid(query=q)
                    .with_limit(k)
                    .do()
                )

                parsed_results = [
                    result[self._weaviate_collection_text_key]
                    for result in results["data"]["Get"][self._weaviate_collection_name]
                ]
            else:
                raise ValueError("Unsupported Weaviate client type")

            passages.extend(dotdict({"long_text": d}) for d in parsed_results)

        return Prediction(passages=passages)

    def get_objects(self, num_samples: int, fields: List[str]) -> List[dict]:
        """Get objects from Weaviate using the cursor API."""
        if self._client_type == "WeaviateClient":
            objects = []
            counter = 0
            for item in self._weaviate_client.data_object.get():
                if counter >= num_samples:
                    break
                new_object = {
                    key: item['properties'][key]
                    for key in item['properties']
                    if key in fields
                }
                objects.append(new_object)
                counter += 1
            return objects
        else:
            raise ValueError(
                "`get_objects` is not supported for the v3 Weaviate Python client, please upgrade to v4."
            )

    def insert(self, new_object_properties: dict):
        if self._client_type == "WeaviateClient":
            self._weaviate_client.data_object.create(
                data_object=new_object_properties,
                class_name=self._weaviate_collection_name,
                uuid=get_valid_uuid(uuid4()),
            )
        else:
            raise AttributeError(
                "`insert` is not supported for the v3 Weaviate Python client, please upgrade to v4."
            )
