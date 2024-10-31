from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
import numpy as np
import torch
from .config import EmbeddingConfig, OutputFormat
from .metrics import SimilarityMetric, cosine_similarity, euclidean_similarity, manhattan_similarity
import hashlib
import logging

logger = logging.getLogger(__name__)

class EmbeddingError(Exception):
    """Base exception for embedding errors."""
    pass

class Embedder(ABC):
    """Abstract base class for embedders."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize embedder."""
        self.config = config or EmbeddingConfig()
        self._cache = {} if self.config.cache_embeddings else None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _convert_output_format(
        self,
        embeddings: List[List[float]],
        output_format: OutputFormat
    ) -> Union[List[List[float]], np.ndarray, torch.Tensor]:
        """Convert embeddings to specified format."""
        try:
            if output_format == OutputFormat.LIST:
                return embeddings
            elif output_format == OutputFormat.ARRAY:
                return np.array(embeddings, dtype=np.float32)
            elif output_format == OutputFormat.TENSOR:
                return torch.tensor(embeddings, dtype=torch.float32)
        except Exception as e:
            raise EmbeddingError(f"Format conversion error: {str(e)}")

    def similarity(
        self,
        text1: str,
        text2: str,
        metric: Union[str, SimilarityMetric] = SimilarityMetric.COSINE
    ) -> float:
        """Calculate similarity between texts."""
        try:
            metric = SimilarityMetric(metric.lower() if isinstance(metric, str) else metric)
            embeddings = self.embed([text1, text2], output_format=OutputFormat.ARRAY)
            v1, v2 = embeddings[0], embeddings[1]

            if metric == SimilarityMetric.COSINE:
                return cosine_similarity(v1, v2)
            elif metric == SimilarityMetric.EUCLIDEAN:
                return euclidean_similarity(v1, v2)
            elif metric == SimilarityMetric.MANHATTAN:
                return manhattan_similarity(v1, v2)
        except Exception as e:
            raise EmbeddingError(f"Similarity calculation error: {str(e)}")

    @abstractmethod
    def embed(
        self,
        texts: Union[str, List[str]],
        output_format: Optional[OutputFormat] = None
    ) -> Union[List[List[float]], np.ndarray, torch.Tensor]:
        """Generate embeddings for texts."""
        pass