from collections import defaultdict
from typing import List, Union  # noqa: UP035

import dspy
from dsp.utils import dotdict

try:
    from pyepsilla import vectordb
except ImportError:
    raise ImportError(  # noqa: B904
        "The 'pyepsilla' extra is required to use EpsillaRM. Install it with `pip install dspy-ai[epsilla]`",
    )


class EpsillaRM(dspy.Retrieve):
    def __init__(
        self,
        epsilla_client: vectordb.Client,
        db_name: str,
        db_path: str,
        table_name: str,
        k: int = 3,
        page_content: str = "document",
    ):
        self._epsilla_client = epsilla_client
        self._epsilla_client.load_db(db_name=db_name, db_path=db_path)
        self._epsilla_client.use_db(db_name=db_name)
        self.page_content = page_content
        self.table_name = table_name

        super().__init__(k=k)

    def forward(self, query_or_queries: Union[str, List[str]], k: Union[int, None] = None, **kwargs) -> dspy.Prediction:  # noqa: ARG002
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]
        limit = k if k else self.k
        all_query_results = []

        for query in queries:
            query_results = self._epsilla_client.query(
                table_name=self.table_name,
                query_text=query,
                limit=limit,
                with_distance=True,
            )
            if query_results[0] == 200:
                all_query_results.append(query_results[1]["result"])

        passages = defaultdict(float)

        for result_dict in all_query_results:
            for result in result_dict:
                passages[result[self.page_content]] += result["@distance"]
        sorted_passages = sorted(passages.items(), key=lambda x: x[1], reverse=False)[:limit]
        return dspy.Prediction(passages=[dotdict({"long_text": passage}) for passage, _ in sorted_passages])
