from collections import defaultdict
from typing import List, Union
import os
import dspy
from dsp.utils import dotdict
import requests
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

try:
    from clarifai.client.search import Search
    from google.protobuf import json_format
except ImportError:
    raise ImportError(
        "Clarifai is not installed. Install it using `pip install clarifai`"
    )


class ClarifaiRM(dspy.Retrieve):
    """
    Retrieval module uses clarifai to return the Top K relevant pasages for the given query.
    """

    def __init__(self,
                 clarifai_user_id: str,
                 clarfiai_app_id: str,
                 clarifai_pat: Optional[str] = None,
                 k: int = 3,
                 
    ):
        self.app_id = clarfiai_app_id
        self.user_id = clarifai_user_id
        self.pat = clarifai_pat if clarifai_pat is not None else os.environ["CLARIFAI_PAT"]
        self.k=k

        super().__init__(k=k)
    
    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int]):

        """Uses clarifai-python SDK search function and retrieves top_k similar passages for given query,
       Args:
            query_or_queries : single query or list of queries
            k : Top K relevant documents to return
       
       Returns:
            passages in format of dotdict 
       
       Examples:
       Below is a code snippet that shows how to use Marqo as the default retriver:
        ```python
        import clarifai

        llm = dspy.Clarifai(model=MODEL_URL, api_key="YOUR CLARIFAI_PAT")
        retriever_model = ClarifaiRM(clarifai_user_id="USER_ID", clarfiai_app_id="APP_ID", clarifai_pat="YOUR CLARIFAI_PAT")
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        k = k if k is not None else self.k
        passages = []
        queries = [q for q in queries if q]

        for query in queries:
            clarifai_search = Search(user_id=self.user_id, app_id=self.app_id, top_k=k, pat=self.pat)
            search_response= clarifai_search.query(ranks=[{"text_raw": query}])

            # Retrieve hits
            hits=[hit for data in search_response for hit in data.hits]
            executor = ThreadPoolExecutor(max_workers=10)

            def retrieve_hits(hits):
                header = {"Authorization": f"Key {self.pat}"}
                request = requests.get(hits.input.data.text.url, headers=header)
                request.encoding = request.apparent_encoding
                requested_text = request.text
                return requested_text
            
            futures = [executor.submit(retrieve_hits, hit) for hit in hits]
            results = [future.result() for future in futures]
            passages=[dotdict({"long_text": d}) for d in results]

            return passages
            








    
