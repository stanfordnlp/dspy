from typing import Union, Optional, List
import numpy as np
import torch
from litellm import embedding
from ..base import Embedder, EmbeddingError
from ..config import EmbeddingConfig, OutputFormat
from .utils import generate_cache_key, validate_api_response

class LiteEmbedder(Embedder):
    """LiteLLM-based embedder implementation."""
    
    def __init__(
        self,
        model: str,
        api_key: str,
        dimensions: Optional[int] = None,
        config: Optional[EmbeddingConfig] = None,
        **kwargs
    ):
        """Initialize LiteEmbedder."""
        super().__init__(config)
        
        if not model or not api_key:
            raise ValueError("Model and API key must be provided")
            
        self.model = model
        self.dimensions = dimensions
        self.embedding_kwargs = {'api_key': api_key, **kwargs}
        self.embedding_kwargs = {k: v for k, v in self.embedding_kwargs.items() if v is not None}
        
        self.logger.info(f"Initialized LiteEmbedder with model: {model}")

    def embed(
        self,
        texts: Union[str, List[str]],
        output_format: Optional[OutputFormat] = None
    ) -> Union[List[List[float]], np.ndarray, torch.Tensor]:
        """Generate embeddings."""
        try:
            # Handle single text
            if isinstance(texts, str):
                texts = [texts]
                is_single = True
            else:
                is_single = False

            # Check cache
            if self.config.cache_embeddings:
                embeddings = self._get_cached_embeddings(texts)
            else:
                embeddings = self._generate_embeddings(texts)

            # Format output
            current_format = output_format or self.config.default_output_format
            formatted = self._convert_output_format(embeddings, OutputFormat(current_format))
            
            return formatted[0] if is_single else formatted
            
        except Exception as e:
            raise EmbeddingError(f"Embedding error: {str(e)}")

    def _get_cached_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Retrieve or generate cached embeddings."""
        cache_keys = [generate_cache_key(text) for text in texts]
        cached = [self._cache.get(key) for key in cache_keys]
        
        if all(emb is not None for emb in cached):
            self.logger.info("Using cached embeddings")
            return cached
            
        embeddings = self._generate_embeddings(texts)
        for key, emb in zip(cache_keys, embeddings):
            self._cache[key] = emb
            
        return embeddings

    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using LiteLLM."""
        try:
            response = embedding(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
                **self.embedding_kwargs
            )
            
            return validate_api_response(response, self.dimensions)
            
        except Exception as e:
            raise EmbeddingError(f"API error: {str(e)}")