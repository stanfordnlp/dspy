"""
Retriever model for LanceDB
Author: Prashant Dixit (@PrashantDixit0)
"""

from typing import List, Union

import backoff
import lancedb.table

from dspy import Retrieve, Prediction
from dspy.dsp.utils.settings import settings
from dspy.dsp.utils import dotdict

try:
    import lancedb
except ImportError:
    lancedb = None

if lancedb is None:
    raise ImportError(
        "The lancedb library is required to use LancedbRM. Install it with `pip install dspy-ai[lancedb]`",
    )


import openai

try:
    OPENAI_LEGACY = int(openai.version.__version__[0]) == 0
except Exception:
    OPENAI_LEGACY = True

try:
    import openai.error
    ERRORS = (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError)
except Exception:
    ERRORS = (openai.RateLimitError, openai.APIError)

    
class LancedbRM(Retrieve):
    """
    A retrieval module that uses LanceDB to return the top passages for a given query.

    Assumes that the LanceDB table has been created and populated with the following metadata:
        - text: The text of the passage

    Args:
        table_name (str): The name of the table to query against.
        persist_directory (str): directory where database is stored.
        k (int, optional): The number of top passages to retrieve. Defaults to 3.

    Returns:
        dspy.Prediction: An object containing the retrieved passages.

    Examples:
        Below is a code snippet that shows how to use this as the default retriever:
        ```python
        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        retriever_model = LancedbRM()
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to use this in the forward() function of a module
        ```python
        self.retrieve = LancedbRM(k=num_passages)
        ```
    """

    def __init__(
        self,
        table_name: str,
        persist_directory: str,
        k: int = 3,
    ):
        
        self._table = self._init_lancedb(
            table_name, persist_directory,
        )

        super().__init__(k=k)

    def _init_lancedb(
        self,
        table_name: str,
        persist_directory: str,
    ):
        """
            Initialize LanceDB database and return the table instance.

            Args:
                table_name (str): The name of the table to load. If the table does not exist, it will be Raise Value Error.

            Raises:
                ValueError: If table is already not created

            Returns:
                Lancedb.table: The loaded table.
        """

        self.db = lancedb.connect(persist_directory)
        table_names = self.db.table_names()

        if table_name not in table_names:
            raise ValueError(
                f"Table does not exist in lancedb database persist_directory. Create table with name {table_name}, then reintialize retriever instance",
            )
        else:
            table = self.db.open_table(table_name)
        return table
    
    
    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=settings.backoff_time,
    )

    def forward(self, query_or_queries: Union[str, List[str]]) -> Prediction:
        """Search with Lancedb for self.k top passages for query

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
        queries = [q for q in queries if q]  # Filter empty queries
        
        passages = []
        for q in queries:
            results = self._table.search(q).limit(self.k).to_list()
            for r in results:
                passages.append(r['text'])
        return Prediction(passages=[dotdict({"long_text": passage}) for passage in passages])
