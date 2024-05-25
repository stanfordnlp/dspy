"""
Retriever model for Pinecone
Author: Dhar Rawal (@drawal1)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import backoff

import dspy
from dsp.utils import dotdict

try:
    import pinecone
except ImportError:
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


class CloudEmbedProvider(ABC):
    def __init__ (self, model, api_key=None):
        self.model = model
        self.api_key = api_key
    
    @abstractmethod
    def get_embeddings(self, queries: List[str]) -> List[List[float]]:
        pass

class OpenAIEmbed(CloudEmbedProvider):
    def __init__(self, model="text-embedding-ada-002", api_key: Optional[str]=None, org: Optional[str]=None):
        super().__init__(model, api_key)
        self.org = org
        if self.api_key:
            openai.api_key = self.api_key
        if self.org:
            openai.organization = org

    
    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=15,
    )
    def get_embeddings(self, queries: List[str]) -> List[List[float]]:
        if OPENAI_LEGACY:
            embedding = openai.Embedding.create(
                input=queries, model=self.model,
            )
        else:
            embedding = openai.embeddings.create(
                input=queries, model=self.model,
            ).model_dump()
        return [embedding["embedding"] for embedding in embedding["data"]]
    
class CohereEmbed(CloudEmbedProvider):
    def __init__(self, model: str = "multilingual-22-12", api_key: Optional[str] = None):
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "The cohere library is required to use CohereEmbed. Install it with `pip install cohere`",
            )
        super().__init__(model, api_key)
        self.client = cohere.Client(api_key)
    
    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_time=15,
    )
    def get_embeddings(self, queries: List[str]) -> List[List[float]]:
        embeddings = self.client.embed(texts=queries, model=self.model).embeddings
        return embeddings
    


class PineconeRM(dspy.Retrieve):
    """
    A retrieval module that uses Pinecone to return the top passages for a given query.

    Assumes that the Pinecone index has been created and populated with the following metadata:
        - text: The text of the passage

    Args:
        pinecone_index_name (str): The name of the Pinecone index to query against.
        pinecone_api_key (str, optional): The Pinecone API key. Defaults to None.
        local_embed_model (str, optional): The local embedding model to use. A popular default is "sentence-transformers/all-mpnet-base-v2".
        cloud_emded_provider (CloudEmbedProvider, optional): The cloud embedding provider to use. Defaults to None.
        k (int, optional): The number of top passages to retrieve. Defaults to 3.

    Returns:
        dspy.Prediction: An object containing the retrieved passages.

    Examples:
        Below is a code snippet that shows how to use this as the default retriver:
        ```python
        llm = dspy.OpenAI(model="gpt-3.5-turbo")
        retriever_model = PineconeRM(index_name, cloud_emded_provider=OpenAIEmbed())
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
        local_embed_model: Optional[str] = None,
        cloud_emded_provider: Optional[CloudEmbedProvider] = None,
        k: int = 3,
    ):
        if local_embed_model is not None:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ModuleNotFoundError(
                "You need to install Hugging Face transformers library to use a local embedding model with PineconeRM.",
            ) from exc

            self._local_embed_model = AutoModel.from_pretrained(local_embed_model)
            self._local_tokenizer = AutoTokenizer.from_pretrained(local_embed_model)
            self.use_local_model = True
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else
                'mps' if torch.backends.mps.is_available()
                else 'cpu',
            )

        elif cloud_emded_provider is not None:
            self.use_local_model = False
            self.cloud_emded_provider = cloud_emded_provider
        
        else:
            raise ValueError(
                "Either local_embed_model or cloud_embed_provider must be provided.",
            )

        if pinecone_api_key is None:
            self.pinecone_client = pinecone.Pinecone()
        else:
            self.pinecone_client = pinecone.Pinecone(api_key=pinecone_api_key)

        self._pinecone_index = self.pinecone_client.Index(pinecone_index_name)

        super().__init__(k=k)

    
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
        if not self.use_local_model:
            return self.cloud_emded_provider.get_embeddings(queries)

        try:
            import torch
        except ImportError as exc:
            raise ModuleNotFoundError(
                "You need to install torch to use a local embedding model with PineconeRM.",
            ) from exc
        
        # Use local model
        encoded_input = self._local_tokenizer(queries, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            model_output = self._local_embed_model(**encoded_input.to(self.device))

        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.cpu().numpy().tolist()

        # we need a pooling strategy to get a single vector representation of the input
        # so the default is to take the mean of the hidden states

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> dspy.Prediction:
        """Search with pinecone for self.k top passages for query

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            k (Optional[int]): The number of top passages to retrieve. Defaults to self.k

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
        embeddings = self._get_embeddings(queries)
        # For single query, just look up the top k passages
        if len(queries) == 1:
            results_dict = self._pinecone_index.query(
                vector=embeddings[0], top_k=k, include_metadata=True,
            )

            # Sort results by score
            sorted_results = sorted(
                results_dict["matches"], key=lambda x: x.get("scores", 0.0), reverse=True,
            )
            
            passages = [result["metadata"]["text"] for result in sorted_results]
            passages = [dotdict({"long_text": passage}) for passage in passages]
            return passages

        # For multiple queries, query each and return the highest scoring passages
        # If a passage is returned multiple times, the score is accumulated. For this reason we increase top_k by 3x
        passage_scores = {}
        for embedding in embeddings:
            results_dict = self._pinecone_index.query(
                vector=embedding, top_k=k * 3, include_metadata=True,
            )
            for result in results_dict["matches"]:
                passage_scores[result["metadata"]["text"]] = (
                    passage_scores.get(result["metadata"]["text"], 0.0)
                    + result["score"]
                )
        
        sorted_passages = sorted(
            passage_scores.items(), key=lambda x: x[1], reverse=True,
        )[: k]
        
        passages=[dotdict({"long_text": passage}) for passage, _ in sorted_passages]
        return passages