from typing import List, Optional, Union

import dspy
from dsp.utils.utils import dotdict

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents._paging import SearchItemPaged
    from azure.search.documents.models import QueryType
except ImportError:
    raise ImportError(
        "You need to install azure-search-documents library"
        "Please use the command: pip install azure-search-documents",
    )

class AzureAISearchRM(dspy.Retrieve):

    """
    A retrieval module that utilizes Azure AI Search to retrieve top passages for a given query.

    Args:
        search_service_name (str): The name of the Azure AI Search service.
        search_api_key (str): The API key for accessing the Azure AI Search service.
        search_index_name (str): The name of the search index in the Azure AI Search service.
        field_text (str): The name of the field containing text content in the search index. This field will be mapped to the "content" field in the dsp framework.
        k (int, optional): The default number of top passages to retrieve. Defaults to 3.
        semantic_ranker (bool, optional): Whether to use semantic ranking. Defaults to False.
        filter (str, optional): Additional filter query. Defaults to None.
        query_language (str, optional): The language of the query. Defaults to "en-Us".
        query_speller (str, optional): The speller mode. Defaults to "lexicon".
        use_semantic_captions (bool, optional): Whether to use semantic captions. Defaults to False.
        query_type (Optional[QueryType], optional): The type of query. Defaults to QueryType.FULL.
        semantic_configuration_name (str, optional): The name of the semantic configuration. Defaults to None.

    Examples:
        Below is a code snippet that demonstrates how to instantiate and use the AzureAISearchRM class:
        ```python
        search_service_name = "your_search_service_name"
        search_api_key = "your_search_api_key"
        search_index_name = "your_search_index_name"
        field_text = "text_content_field"

        azure_search_retriever = AzureAISearchRM(search_service_name, search_api_key, search_index_name, field_text)
        ```

    Attributes:
        search_service_name (str): The name of the Azure AI Search service.
        search_api_key (str): The API key for accessing the Azure AI Search service.
        search_index_name (str): The name of the search index in the Azure AI Search service.
        field_text (str): The name of the field containing text content in the search index.
        endpoint (str): The endpoint URL for the Azure AI Search service.
        credential (AzureKeyCredential): The Azure key credential for accessing the service.
        client (SearchClient): The Azure AI Search client instance.

    Methods:
        forward(query_or_queries: Union[str, List[str]], k: Optional[int]) -> dspy.Prediction:
            Search for the top passages corresponding to the given query or queries.

        azure_search_request(
            key_content: str,
            client: SearchClient,
            query: str,
            top: int,
            semantic_ranker: bool,
            filter: str,
            query_language: str,
            query_speller: str,
            use_semantic_captions: bool,
            query_type: QueryType,
            semantic_configuration_name: str
        ) -> List[dict]:
            Perform a search request to the Azure AI Search service.

        process_azure_result(results: SearchItemPaged, content_key: str, content_score: str) -> List[dict]:
            Process the results received from the Azure AI Search service and map them to the correct format.

    Raises:
        ImportError: If the required Azure AI Search libraries are not installed.

    Note:
        This class relies on the 'azure-search-documents' library for interacting with the Azure AI Search service.
        Ensure that you have the necessary permissions and correct configurations set up in Azure before using this class.
    """

    def __init__(
        self,
        search_service_name: str,
        search_api_key: str,
        search_index_name: str,
        field_text: str,
        k: int = 3,
        semantic_ranker: bool = False,
        filter: str = None,
        query_language: str = "en-Us",
        query_speller: str = "lexicon",
        use_semantic_captions: bool = False,
        query_type: Optional[QueryType] = QueryType.FULL,
        semantic_configuration_name: str = None,

    ):
        self.search_service_name = search_service_name
        self.search_api_key = search_api_key
        self.search_index_name = search_index_name
        self.endpoint=f"https://{self.search_service_name}.search.windows.net"
        self.field_text = field_text # field name of the text content
        # Create a client
        self.credential = AzureKeyCredential(self.search_api_key)
        self.client = SearchClient(endpoint=self.endpoint,
                        index_name=self.search_index_name,
                        credential=self.credential)
        self.semantic_ranker = semantic_ranker
        self.filter = filter
        self.query_language = query_language
        self.query_speller = query_speller
        self.use_semantic_captions = use_semantic_captions
        self.query_type = query_type
        self.semantic_configuration_name = semantic_configuration_name

        super().__init__(k=k)

    def azure_search_request(self,key_content: str,  client: SearchClient, query: str, top: int, semantic_ranker: bool, filter: str, query_language: str, query_speller: str, use_semantic_captions: bool, query_type: QueryType, semantic_configuration_name: str):
        """
        Search in Azure AI Search Index
        """

        # TODO: Add Support for Vector Search And Hybride Search
        if semantic_ranker:
            results = client.search(search_text=query,
                                    filter=filter,
                                    query_type=query_type,
                                    query_language = query_language,
                                    query_speller=query_speller,
                                    semantic_configuration_name=semantic_configuration_name,
                                    top=top,
                                    query_caption = (
                                        'extractive|highlight-false'
                                        if use_semantic_captions
                                        else None
                                    ),
                                )
        else:
            results = client.search(search_text=query,top=top,filter=filter)

        sorted_results = sorted(results, key=lambda x: x['@search.score'], reverse=True)

        sorted_results = self.process_azure_result(sorted_results, key_content, key_content)

        return sorted_results

    def process_azure_result(self,results:SearchItemPaged, content_key:str, content_score: str):
        """
        process received result from Azure AI Search as dictionary array and map content and score to correct format
        """
        res = []
        for result in results:
            tmp = {}
            for key, value in result.items():
                if(key == content_key):
                    tmp["text"] = value # assign content
                elif(key == content_score):
                    tmp["score"] = value
                else:
                    tmp[key] = value
            res.append(tmp)
        return res

    def forward(self, query_or_queries: Union[str, List[str]],  k: Optional[int]) -> dspy.Prediction:
        """
        Search with pinecone for self.k top passages for query

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """

        k = k if k is not None else self.k
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        queries = [q for q in queries if q]  # Filter empty queries

        passages = []
        for query in queries:
            results = self.azure_search_request(self.field_text,
                                                self.client, query,
                                                k,
                                                self.semantic_ranker,
                                                self.filter,
                                                self.query_language,
                                                self.query_speller,
                                                self.use_semantic_captions,
                                                self.query_type,
                                                self.semantic_configuration_name)
            passages.extend(dotdict({"long_text": d['text']}) for d in results)

        return passages
