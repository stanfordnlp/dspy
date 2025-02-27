from typing import Optional, Union

import dspy
from dspy.dsp.utils.utils import dotdict

try:
    from ragatouille import RAGPretrainedModel
except ImportError:
    raise Exception(
        "You need to install RAGAtouille library, Please use the command: pip install ragatouille",
    )


class RAGatouilleRM(dspy.Retrieve):
    """A retrieval model that uses RAGatouille library to return the top passages for a given query.

    Assumes that you already have an index created with RAGatouille.
    Reference: https://github.com/bclavie/RAGatouille

    Args:
        index_root (str): Folder path where you index is stored.
        index_name (str): Name of the index you want to retrieve from.
        k (int, optional): The default number of passages to retrieve. Defaults to 3.

    Examples:
        Below is a code snippet that shows how to use RAGatouille index as the default retriver:
        ```python
        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        rm = RAGatouilleRM(index_root="ragatouille/colbert/indexes", index_name="my_index")
        dspy.settings.configure(lm=llm, rm=rm)
        ```
    """

    def __init__(self, index_root: str, index_name: str, k: int = 3):
        super().__init__(k=k)
        self.index_path = f"{index_root}/{index_name}"
        try:
            self.model = RAGPretrainedModel.from_index(index_path=self.index_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Index not found: {e}")

    def forward(self, query_or_queries: Union[str, list[str]], k: Optional[int]) -> dspy.Prediction:
        """Search with RAGAtouille based index for self.k top passages for query

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
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
            results = self.model.search(query=query, k=k)
            passages.extend(dotdict({"long_text": d["content"]}) for d in results)
        return dspy.Prediction(passages=passages)