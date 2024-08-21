"""
Retriever model for Pinecone
Author: Dhar Rawal (@drawal1)
"""

from typing import List, Optional, Union

import backoff

from dspy import Retrieve, Prediction
from dsp.utils.settings import settings
from dsp.utils import dotdict

try:
    import pinecone
except ImportError:
    pinecone = None

if pinecone is None:
    raise ImportError(
        "The pinecone library is required to use PineconeRM. Install it with `pip install dspy-ai[pinecone]`",
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

class PineconeRM(Retrieve):
    """
    A retrieval module that uses Pinecone to return the top passages for a given query.

    Assumes that the Pinecone index has been created and populated with the following metadata:
        - text: The text of the passage

    Args:
        pinecone_index_name (str): The name of the Pinecone index to query against.
        pinecone_api_key (str, optional): The Pinecone API key. Defaults to None.
        pinecone_env (str, optional): The Pinecone environment. Defaults to None.
        local_embed_model (str, optional): The local embedding model to use. A popular default is "sentence-transformers/all-mpnet-base-v2".
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
        retriever_model = PineconeRM(openai.api_key)
        dspy.settings.configure(lm=llm, rm=retriever_model)
        ```

        Below is a code snippet that shows how to use this in the forward() function of a module
        ```python
        self.retrieve = PineconeRM(k=num_passages)
        ```
    """

    def __init__(
        self,
        pinecone_index_name: str,
        pinecone_api_key: Optional[str] = None,
        pinecone_env: Optional[str] = None,
        local_embed_model: Optional[str] = None,
        openai_embed_model: Optional[str] = "text-embedding-ada-002",
        openai_api_key: Optional[str] = None,
        openai_org: Optional[str] = None,
        k: int = 3,
    ):
        if local_embed_model is not None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ModuleNotFoundError(
                "You need to install Hugging Face transformers (with torch dependencies - pip install transformers[torch]) library to use a local embedding model with PineconeRM.",
            ) from exc

            self._local_embed_model = AutoModel.from_pretrained(local_embed_model)
            self._local_tokenizer = AutoTokenizer.from_pretrained(local_embed_model)
            self.use_local_model = True
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else
                'mps' if torch.backends.mps.is_available()
                else 'cpu',
            )
        elif openai_embed_model is not None:
            self._openai_embed_model = openai_embed_model
            self.use_local_model = False
            # If not provided, defaults to env vars OPENAI_API_KEY and OPENAI_ORGANIZATION
            if openai_api_key:
                openai.api_key = openai_api_key
            if openai_org:
                openai.organization = openai_org
        else:
            raise ValueError(
                "Either local_embed_model or openai_embed_model must be provided.",
            )

        self._pinecone_index = self._init_pinecone(
            pinecone_index_name, pinecone_api_key, pinecone_env,
        )

        super().__init__(k=k)

    def _init_pinecone(
        self,
        index_name: str,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        dimension: Optional[int] = None,
        distance_metric: Optional[str] = None,
    ) -> pinecone.Index:
        """Initialize pinecone and return the loaded index.

        Args:
            index_name (str): The name of the index to load. If the index is not does not exist, it will be created.
            api_key (str, optional): The Pinecone API key, defaults to env var PINECONE_API_KEY if not provided.
            environment (str, optional): The environment (ie. `us-west1-gcp` or `gcp-starter`. Defaults to env PINECONE_ENVIRONMENT.

        Raises:
            ValueError: If api_key or environment is not provided and not set as an environment variable.

        Returns:
            pinecone.Index: The loaded index.
        """

        # Pinecone init overrides default if kwargs are present, so we need to exclude if None
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if environment:
            kwargs["environment"] = environment
        pinecone.init(**kwargs)

        active_indexes = pinecone.list_indexes()
        if index_name not in active_indexes:
            if dimension is None and distance_metric is None:
                raise ValueError(
                    "dimension and distance_metric must be provided since the index provided does not exist.",
                )

            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric=distance_metric,
            )

        return pinecone.Index(index_name)
    
    def _mean_pooling(
            self, 
            model_output, 
            attention_mask,
        ):
        try:
            import torch
        except ImportError as exc:
            raise ModuleNotFoundError(
                "You need to install torch to use a local embedding model with PineconeRM.",
            ) from exc

        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
 
    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=settings.backoff_time,
    )
    def _get_embeddings(
        self, 
        queries: List[str],
    ) -> List[List[float]]:
        """Return query vector after creating embedding using OpenAI

        Args:
            queries (list): List of query strings to embed.

        Returns:
            List[List[float]]: List of embeddings corresponding to each query.
        """
        try:
            import torch
        except ImportError as exc:
            raise ModuleNotFoundError(
                "You need to install torch to use a local embedding model with PineconeRM.",
            ) from exc
        
        if not self.use_local_model:
            if OPENAI_LEGACY:
                embedding = openai.Embedding.create(
                    input=queries, model=self._openai_embed_model,
                )
            else:
                embedding = openai.embeddings.create(
                    input=queries, model=self._openai_embed_model,
                ).model_dump()
            return [embedding["embedding"] for embedding in embedding["data"]]
        
        # Use local model
        encoded_input = self._local_tokenizer(queries, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            model_output = self._local_embed_model(**encoded_input.to(self.device))

        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.cpu().numpy().tolist()

        # we need a pooling strategy to get a single vector representation of the input
        # so the default is to take the mean of the hidden states

    def forward(self, query_or_queries: Union[str, List[str]]) -> Prediction:
        """Search with pinecone for self.k top passages for query

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

        # For single query, just look up the top k passages
        if len(queries) == 1:
            results_dict = self._pinecone_index.query(
                embeddings[0], top_k=self.k, include_metadata=True,
            )

            # Sort results by score
            sorted_results = sorted(
                results_dict["matches"], key=lambda x: x.get("scores", 0.0), reverse=True,
            )
            passages = [result["metadata"]["text"] for result in sorted_results]
            passages = [dotdict({"long_text": passage for passage in passages})]
            return Prediction(passages=passages)

        # For multiple queries, query each and return the highest scoring passages
        # If a passage is returned multiple times, the score is accumulated. For this reason we increase top_k by 3x
        passage_scores = {}
        for embedding in embeddings:
            results_dict = self._pinecone_index.query(
                embedding, top_k=self.k * 3, include_metadata=True,
            )
            for result in results_dict["matches"]:
                passage_scores[result["metadata"]["text"]] = (
                    passage_scores.get(result["metadata"]["text"], 0.0)
                    + result["score"]
                )

        sorted_passages = sorted(
            passage_scores.items(), key=lambda x: x[1], reverse=True,
        )[: self.k]
        return Prediction(passages=[dotdict({"long_text": passage}) for passage, _ in sorted_passages])
