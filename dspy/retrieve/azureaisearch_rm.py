"""
Retriever module for Azure AI Search
Author: Prajapati Harishkumar Kishorkumar (@HARISHKUMAR1112001)
"""

import warnings
from typing import Any, Callable, List, Optional, Union

import dspy
from dsp.utils.utils import dotdict

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents._paging import SearchItemPaged
    from azure.search.documents.models import QueryType, VectorFilterMode, VectorizedQuery
except ImportError:
    raise ImportError(
        "You need to install azure-search-documents library"
        "Please use the command: pip install azure-search-documents==11.6.0b1",
    )

try:
    import openai
except ImportError:
    warnings.warn(
        "`openai` is not installed. Install it with `pip install openai` to use AzureOpenAI embedding models.",
        category=ImportWarning,
    )


class AzureAISearchRM(dspy.Retrieve):

    """
    A retrieval module that utilizes Azure AI Search to retrieve top passages for a given query.

    Args:
        search_service_name (str): The name of the Azure AI Search service.
        search_api_key (str): The API key for accessing the Azure AI Search service.
        search_index_name (str): The name of the search index in the Azure AI Search service.
        field_text (str): The name of the field containing text content in the search index. This field will be mapped to the "content" field in the dsp framework.
        field_vector (Optional[str]): The name of the field containing vector content in the search index. Defaults to None.
        k (int, optional): The default number of top passages to retrieve. Defaults to 3.
        azure_openai_client (Optional[openai.AzureOpenAI]): An instance of the AzureOpenAI client. Either openai_client or embedding_func must be provided. Defaults to None.
        openai_embed_model (Optional[str]): The name of the OpenAI embedding model. Defaults to "text-embedding-ada-002".
        embedding_func (Optional[Callable]): A function for generating embeddings. Either openai_client or embedding_func must be provided. Defaults to None.
        semantic_ranker (bool, optional): Whether to use semantic ranking. Defaults to False.
        filter (str, optional): Additional filter query. Defaults to None.
        query_language (str, optional): The language of the query. Defaults to "en-Us".
        query_speller (str, optional): The speller mode. Defaults to "lexicon".
        use_semantic_captions (bool, optional): Whether to use semantic captions. Defaults to False.
        query_type (Optional[QueryType], optional): The type of query. Defaults to QueryType.FULL.
        semantic_configuration_name (str, optional): The name of the semantic configuration. Defaults to None.
        is_vector_search (Optional[bool]): Whether to enable vector search. Defaults to False.
        is_hybrid_search (Optional[bool]): Whether to enable hybrid search. Defaults to False.
        is_fulltext_search (Optional[bool]): Whether to enable fulltext search. Defaults to True.
        vector_filter_mode (Optional[VectorFilterMode]): The vector filter mode. Defaults to None.

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
        endpoint (str): The endpoint URL for the Azure AI Search service.
        field_text (str): The name of the field containing text content in the search index.
        field_vector (Optional[str]): The name of the field containing vector content in the search index.
        azure_openai_client (Optional[openai.AzureOpenAI]): An instance of the AzureOpenAI client.
        openai_embed_model (Optional[str]): The name of the OpenAI embedding model.
        embedding_func (Optional[Callable]): A function for generating embeddings.
        credential (AzureKeyCredential): The Azure key credential for accessing the service.
        client (SearchClient): The Azure AI Search client instance.
        semantic_ranker (bool): Whether to use semantic ranking.
        filter (str): Additional filter query.
        query_language (str): The language of the query.
        query_speller (str): The speller mode.
        use_semantic_captions (bool): Whether to use semantic captions.
        query_type (Optional[QueryType]): The type of query.
        semantic_configuration_name (str): The name of the semantic configuration.
        is_vector_search (Optional[bool]): Whether to enable vector search.
        is_hybrid_search (Optional[bool]): Whether to enable hybrid search.
        is_fulltext_search (Optional[bool]): Whether to enable fulltext search.
        vector_filter_mode (Optional[VectorFilterMode]): The vector filter mode.

    Methods:
        forward(query_or_queries: Union[str, List[str]], k: Optional[int]) -> dspy.Prediction:
            Search for the top passages corresponding to the given query or queries.

        azure_search_request(
            self,
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
            semantic_configuration_name: str,
            is_vector_search: bool,
            is_hybrid_search: bool,
            is_fulltext_search: bool,
            field_vector: str,
            vector_filter_mode: VectorFilterMode
        ) -> List[dict]:
            Perform a search request to the Azure AI Search service.

        process_azure_result(
            self,
            results:SearchItemPaged,
            content_key:str,
            content_score: str
        ) -> List[dict]:
            Process the results received from the Azure AI Search service and map them to the correct format.

        get_embeddings(
            self,
            query: str,
            k_nearest_neighbors: int,
            field_vector: str
        ) -> List | Any:
            Returns embeddings for the given query.

        check_semantic_configuration(
            self,
            semantic_configuration_name,
            query_type
        ):
            Checks semantic configuration.

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
        field_vector: Optional[str] = None,
        k: int = 3,
        azure_openai_client: Optional[openai.AzureOpenAI] = None,
        openai_embed_model: Optional[str] = "text-embedding-ada-002",
        embedding_func: Optional[Callable] = None,
        semantic_ranker: bool = False,
        filter: str = None,
        query_language: str = "en-Us",
        query_speller: str = "lexicon",
        use_semantic_captions: bool = False,
        query_type: Optional[QueryType] = QueryType.FULL,
        semantic_configuration_name: str = None,
        is_vector_search: Optional[bool] = False,
        is_hybrid_search: Optional[bool] = False,
        is_fulltext_search: Optional[bool] = True,
        vector_filter_mode: Optional[VectorFilterMode.PRE_FILTER] = None,
    ):
        self.search_service_name = search_service_name
        self.search_api_key = search_api_key
        self.search_index_name = search_index_name
        self.endpoint = f"https://{self.search_service_name}.search.windows.net"
        self.field_text = field_text  # field name of the text content
        self.field_vector = field_vector  # field name of the vector content
        self.azure_openai_client = azure_openai_client
        self.openai_embed_model = openai_embed_model
        self.embedding_func = embedding_func
        # Create a client
        self.credential = AzureKeyCredential(self.search_api_key)
        self.client = SearchClient(
            endpoint=self.endpoint, index_name=self.search_index_name, credential=self.credential,
        )
        self.semantic_ranker = semantic_ranker
        self.filter = filter
        self.query_language = query_language
        self.query_speller = query_speller
        self.use_semantic_captions = use_semantic_captions
        self.query_type = query_type
        self.semantic_configuration_name = semantic_configuration_name
        self.is_vector_search = is_vector_search
        self.is_hybrid_search = is_hybrid_search
        self.is_fulltext_search = is_fulltext_search
        self.vector_filter_mode = vector_filter_mode

        super().__init__(k=k)

    def azure_search_request(
        self,
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
        semantic_configuration_name: str,
        is_vector_search: bool,
        is_hybrid_search: bool,
        is_fulltext_search: bool,
        field_vector: str,
        vector_filter_mode: VectorFilterMode,
    ):
        """
        Search in Azure AI Search Index
        """

        if is_vector_search:
            vector_query = self.get_embeddings(query, top, field_vector)
            if semantic_ranker:
                self.check_semantic_configuration(semantic_configuration_name, query_type)
                results = client.search(
                    search_text=None,
                    filter=filter,
                    query_type=query_type,
                    vector_queries=[vector_query],
                    vector_filter_mode=vector_filter_mode,
                    semantic_configuration_name=semantic_configuration_name,
                    top=top,
                    query_caption=("extractive|highlight-false" if use_semantic_captions else None),
                )
            else:
                results = client.search(
                    search_text=None,
                    filter=filter,
                    vector_queries=[vector_query],
                    vector_filter_mode=vector_filter_mode,
                    top=top,
                )
        if is_hybrid_search:
            vector_query = self.get_embeddings(query, top, field_vector)
            if semantic_ranker:
                self.check_semantic_configuration(semantic_configuration_name, query_type)
                results = client.search(
                    search_text=query,
                    filter=filter,
                    query_type=query_type,
                    query_language=query_language,
                    query_speller=query_speller,
                    semantic_configuration_name=semantic_configuration_name,
                    top=top,
                    vector_queries=[vector_query],
                    vector_filter_mode=vector_filter_mode,
                    query_caption=("extractive|highlight-false" if use_semantic_captions else None),
                )
            else:
                results = client.search(
                    search_text=query,
                    filter=filter,
                    query_language=query_language,
                    query_speller=query_speller,
                    top=top,
                    vector_queries=[vector_query],
                    vector_filter_mode=vector_filter_mode,
                )
        if is_fulltext_search:
            if semantic_ranker:
                self.check_semantic_configuration(semantic_configuration_name, query_type)
                results = client.search(
                    search_text=query,
                    filter=filter,
                    query_type=query_type,
                    query_language=query_language,
                    query_speller=query_speller,
                    semantic_configuration_name=semantic_configuration_name,
                    top=top,
                    query_caption=("extractive|highlight-false" if use_semantic_captions else None),
                )
            else:
                results = client.search(search_text=query, top=top, filter=filter)

        sorted_results = sorted(results, key=lambda x: x["@search.score"], reverse=True)

        sorted_results = self.process_azure_result(sorted_results, key_content, key_content)

        return sorted_results

    def process_azure_result(self, results: SearchItemPaged, content_key: str, content_score: str):
        """
        process received result from Azure AI Search as dictionary array and map content and score to correct format
        """
        res = []
        for result in results:
            tmp = {}
            for key, value in result.items():
                if key == content_key:
                    tmp["text"] = value  # assign content
                elif key == content_score:
                    tmp["score"] = value
                else:
                    tmp[key] = value
            res.append(tmp)
        return res

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int]) -> dspy.Prediction:
        """
        Search with pinecone for self.k top passages for query

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.

        Returns:
            dspy.Prediction: An object containing the retrieved passages.
        """

        k = k if k is not None else self.k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        queries = [q for q in queries if q]  # Filter empty queries

        passages = []
        for query in queries:
            results = self.azure_search_request(
                self.field_text,
                self.client,
                query,
                k,
                self.semantic_ranker,
                self.filter,
                self.query_language,
                self.query_speller,
                self.use_semantic_captions,
                self.query_type,
                self.semantic_configuration_name,
                self.is_vector_search,
                self.is_hybrid_search,
                self.is_fulltext_search,
                self.field_vector,
                self.vector_filter_mode,
            )
            passages.extend(dotdict({"long_text": d["text"]}) for d in results)

        return passages

    def get_embeddings(self, query: str, k_nearest_neighbors: int, field_vector: str) -> List | Any:
        """
        Returns embeddings for the given query.

        Args:
            query (str): The query for which embeddings are to be retrieved.
            k_nearest_neighbors (int): The number of nearest neighbors to consider.
            field_vector (str): The field vector to use for embeddings.

        Returns:
            list: A list containing the vectorized query.
            Any: The result of embedding_func if azure_openai_client is not provided.

        Raises:
            AssertionError: If neither azure_openai_client nor embedding_func is provided,
                or if field_vector is not provided.
        """
        assert (
            self.azure_openai_client or self.embedding_func
        ), "Either azure_openai_client or embedding_func must be provided."
        
        if self.azure_openai_client is not None:
            assert field_vector, "field_vector must be provided."
            
            embedding = (
                self.azure_openai_client.embeddings.create(input=query, model=self.openai_embed_model).data[0].embedding
            )
            vector_query = VectorizedQuery(
                vector=embedding, k_nearest_neighbors=k_nearest_neighbors, fields=field_vector,
            )
            return [vector_query]
        else:
            return self.embedding_func(query)

    def check_semantic_configuration(self, semantic_configuration_name, query_type):
        """
        Checks semantic configuration.

        Args:
            semantic_configuration_name: The name of the semantic configuration.
            query_type: The type of the query.

        Raises:
            AssertionError: If semantic_configuration_name is not provided
                or if query_type is not QueryType.SEMANTIC.
        """
        assert semantic_configuration_name, "Semantic configuration name must be provided."
        assert query_type == QueryType.SEMANTIC, "Query type must be QueryType.SEMANTIC."
