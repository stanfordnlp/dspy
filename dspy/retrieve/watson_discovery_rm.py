import json
from typing import Optional, Union

import requests
from requests.auth import HTTPBasicAuth

import dspy
from dsp.utils import dotdict


class WatsonDiscoveryRM(dspy.Retrieve):
    """A retrieval module that uses Watson Discovery to return the top passages for a given query.

    Args:
        apikey (str): apikey for authentication purposes,
        url (str): endpoint URL that includes the service instance ID
        version (str): Release date of the version of the API you want to use. Specify dates in YYYY-MM-DD format.
        project_id (str): The Universally Unique Identifier (UUID) of the project.
        collection_ids (list): An array containing the collections on which the search will be executed.
        k (int, optional): The number of top passages to retrieve. Defaults to 5.

    Returns:
        dspy.Prediction: An object containing the retrieved passages.
    """

    def __init__(
        self,
        apikey: str,
        url:str,
        version:str,
        project_id:str,
        collection_ids:list=None,
        k: int = 5,
    ):
        if collection_ids is None:
            collection_ids = []
        self.apikey=apikey
        self.url=url,
        self.version:version
        self.project_id:project_id
        self.collection_ids=collection_ids
        self.k: k
        self.query_url=url + "/v2/projects/" + project_id + "/query?version=" + version
        
        super().__init__(k=k)

    def forward(self, query_or_queries: Union[str, list[str]], k: Optional[int]= None) -> dspy.Prediction:
        """Search with Watson Discovery for self.k top passages for query.

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            k (int, optional): The number of top passages to retrieve.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]  # Filter empty queries
        k = self.k if k is None else k

        response=[]
        try:
            for query in queries:
                payload = json.dumps({
                    "collection_ids": self.collection_ids,
                    "natural_language_query": "text:"+query,
                    "passages": {
                        "enabled": True,
                        "count": k,
                        "fields":["text"],
                        "characters": 500,
                        "find_answers": False,
                        "per_document": False,
                    },
                    "highlight": False,
                })
                headers = {'Content-Type': 'application/json'}

                discovery_results = requests.request(
                    "POST",
                    url=self.query_url, 
                    headers=headers,
                    auth=HTTPBasicAuth("apikey", self.apikey),
                    data=payload,
                )
                
                discovery_results.raise_for_status()

                doc_dict={}
                for d in discovery_results.json()["results"]:
                    doc_dict[d["document_id"]]=d["title"]

                for d in discovery_results.json()["passages"]:
                    response.append(dotdict({
                        "title":doc_dict[d["document_id"]],
                        "long_text": doc_dict[d["document_id"]]+" | " + d["passage_text"],
                        "passage_score": d["passage_score"],
                        "document_id": d["document_id"],
                        "collection_id": d["collection_id"],
                        "start_offset": d["start_offset"],
                        "end_offset": d["end_offset"],
                        "field": d["field"],
                    }))

        except requests.exceptions.RequestException as err:
            raise SystemExit(err) from err
        return response