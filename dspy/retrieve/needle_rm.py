from typing import List, Optional, Union

import dspy
from dspy.dsp.utils import dotdict

try:
    from needle.v1 import NeedleClient
except ImportError as err:
    raise ImportError(
        "The 'needle' extra is required to use NeedleRM. Install it with `pip install dspy-ai[needle]`",
    ) from err


class NeedleRM(dspy.Retrieve):
    """
    Retrieval module that uses Needle to retrieve top chunks for a given query from a Needle collection.
    You can find your collection id in the Needle collection page and genrate an API key in your account settings page.

    Args:
        collection_id (str): The id of the collection to retrieve from.
        api_key (Optional[str]): The API key to use for the Needle client. If not provided, value from the environment variable NEEDLE_API_KEY is used.
        k (Optional[int]): The number of chunks to retrieve. If not provided, the default value is 10.
    """

    def __init__(
        self, collection_id: str, api_key: Optional[str] = None, k: Optional[int] = None
    ):
        k = k or 10
        super().__init__(k=k)
        self.collection_id = collection_id
        self.ndl = NeedleClient(api_key=api_key) if api_key else NeedleClient()
        self.k = k

    def forward(
        self,
        query: Union[str, List[str]],
        k: Optional[int] = None,
        **kwargs,
    ) -> dspy.Prediction:
        queries = [query] if isinstance(query, str) else query

        results = []
        for q in queries:
            results.extend(
                self.ndl.collections.search(
                    collection_id=self.collection_id,
                    text=q,
                    top_k=(k or self.k),
                    **kwargs,
                )
            )

        passages = [dotdict({"long_text": r.content}) for r in results]
        return passages
