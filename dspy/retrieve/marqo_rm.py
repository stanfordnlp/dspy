from collections import defaultdict
from typing import List, Union
import dspy

try:
    import marqo
except ImportError:
    raise ImportError(
        "The 'marqo' extra is required to use MarqoRM. Install it with `pip install dspy-ai[marqo]`"
    )

class MarqoRM(dspy.Retrieve):
    """
    A retrieval module that uses Marqo to return the top passages for a given query.

    Assumes that a Marqo index has been created and populated with the following payload:
        - document: The text of the passage

    Args:
        marqo_index_name (str): The name of the marqo index.
        marqo_client (marqo.client.Client): A marqo client instance.
        k (int, optional): The number of top passages to retrieve. Defaults to 3.

    Examples:
        Below is a code snippet that shows how to use Marqo as the default retriver:
        ```python
        import marqo
        marqo_client = marqo.Client(url="http://0.0.0.0:8882")

        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        retriever_model = MarqoRM("my_index_name", marqo_client=marqo_client)
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to use Marqo in the forward() function of a module
        ```python
        self.retrieve = MarqoRM("my_index_name", marqo_client=marqo_client, k=num_passages)
        ```
    """

    def __init__(
        self,
        marqo_index_name: str,
        marqo_client: marqo.client.Client,
        k: int = 3,
    ):
        self._marqo_index_name = marqo_index_name
        self._marqo_client = marqo_client

        super().__init__(k=k)

    def forward(self, query_or_queries: Union[str, List[str]]) -> dspy.Prediction:
        """Search with Marqo for self.k top passages for query

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [q for q in queries if q]  

        all_query_results = []
        for query in queries:
            _result = self._marqo_client.index(self._marqo_index_name).search(
                q=query,
                limit=self.k
            )
            all_query_results.append(_result)

        passages = defaultdict(float)

        for result_dict in all_query_results:
            for result in result_dict['hits']:
                passages[result['document']] += result['_score']

        sorted_passages = sorted(
            passages.items(), key=lambda x: x[1], reverse=True)[:self.k]
        return dspy.Prediction(passages=[passage for passage, _ in sorted_passages])
