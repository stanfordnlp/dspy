"""
Retriever model for chromadb
"""

from typing import Optional, List, Union
import openai
import dspy
import backoff
from dsp.utils import dotdict

try:
    import openai.error
    ERRORS = (
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
        openai.error.APIError,
    )
except Exception:
    ERRORS = (openai.RateLimitError, openai.APIError)

try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None

try:
    import torch
    from transformers import AutoModel, AutoTokenizer
except ImportError as exc:
    raise ModuleNotFoundError(
        "You need to install Hugging Face transformers library to use a local embedding model with ChromadbRM."
    ) from exc

if chromadb is None:
    raise ImportError(
        "The chromadb library is required to use ChromadbRM. Install it with `pip install dspy-ai[chromadb]`"
    )

class ChromadbRM(dspy.Retrieve):
    """
    A retrieval module that uses chromadb to return the top passages for a given query.

    Assumes that the chromadb index has been created and populated with the following metadata:
        - documents: The text of the passage

    Args:
        collection_name (str): chromadb collection name
        persist_directory (str): chromadb persist directory
        openai_embed_model (str, optional): The OpenAI embedding model to use. Defaults to "text-embedding-ada-002".
        openai_api_key (str, optional): The API key for OpenAI. Defaults to None.
        openai_org (str, optional): The organization for OpenAI. Defaults to None.
        k (int, optional): The number of top passages to retrieve. Defaults to 3.

    Returns:
        dspy.Prediction: An object containing the retrieved passages.

    Examples:
        Below is a code snippet that shows how to use this as the default retriever:
        ```python
        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        retriever_model = ChromadbRM('collection_name', 'db_path')
        dspy.settings.configure(lm=llm, rm=retriever_model)
        # to test the retriever with "my query"
        retriever_model("my query")
        ```

        Below is a code snippet that shows how to use this in the forward() function of a module
        ```python
        self.retrieve = ChromadbRM('collection_name', 'db_path', k=num_passages)
        ```
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        openai_embed_model: str = "text-embedding-ada-002",
        openai_api_provider: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_api_type: Optional[str] = None,
        openai_api_base: Optional[str] = None,
        openai_api_version: Optional[str] = None,
        local_embed_model: Optional[str] = None,
        k: int = 7,
    ):
        self._openai_embed_model = openai_embed_model

        self._init_chromadb(collection_name, persist_directory)

        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key,
            api_base=openai_api_base,
            api_type=openai_api_type,
            api_version=openai_api_version,
            model_name=openai_embed_model,
        )

        if local_embed_model is not None:
            self._local_embed_model = AutoModel.from_pretrained(local_embed_model)
            self._local_tokenizer = AutoTokenizer.from_pretrained(local_embed_model)
            self.use_local_model = True
            self.device = torch.device(
                "cuda:0"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.use_local_model = False

        super().__init__(k=k)

    def _init_chromadb(
        self,
        collection_name: str,
        persist_directory: str,
    ) -> chromadb.Collection:
        """Initialize chromadb and return the loaded index.

        Args:
            collection_name (str): chromadb collection name
            persist_directory (str): chromadb persist directory


        Returns:
        """

        self._chromadb_client = chromadb.Client(
            Settings(
                persist_directory=persist_directory,
                is_persistent=True,
            )
        )
        self._chromadb_collection = self._chromadb_client.get_or_create_collection(
            name=collection_name,
        )

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=15,
    )
    def _get_embeddings(self, queries: List[str]) -> List[List[float]]:
        """Return query vector after creating embedding using OpenAI

        Args:
            queries (list): List of query strings to embed.

        Returns:
            List[List[float]]: List of embeddings corresponding to each query.
        """

        if not self.use_local_model:
            # Using OpenAI's embedding model
            embedding = openai.Embedding.create(
                input=queries, model=self._openai_embed_model
            )
            return [embedding.embedding for embedding in embedding.data]

        # Use local model for embeddings
        encoded_input = self._local_tokenizer(
            queries, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            model_output = self._local_embed_model(**encoded_input.to(self.device))

        embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.cpu().numpy().tolist()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(
        self, query_or_queries: Union[str, List[str]], k: Optional[int] = None
    ) -> dspy.Prediction:
        """Search with db for self.k top passages for query

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
        embeddings = self._get_embeddings(queries)

        k = self.k if k is None else k
        results = self._chromadb_collection.query(
            query_embeddings=embeddings, n_results=k
        )

        passages = [dotdict({"long_text": x}) for x in results["documents"][0]]

        return passages
