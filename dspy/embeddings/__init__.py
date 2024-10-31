# File: dspy/embeddings/__init__.py

from .base import Embedder, EmbeddingError
from .config import EmbeddingConfig, OutputFormat
from .metrics import SimilarityMetric
from .providers.lite_embedder import LiteEmbedder

__all__ = [
    'Embedder',
    'EmbeddingError',
    'EmbeddingConfig',
    'OutputFormat',
    'SimilarityMetric',
    'LiteEmbedder'
]