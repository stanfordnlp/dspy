from dataclasses import dataclass
from enum import Enum
from typing import Optional

class OutputFormat(str, Enum):
    """Supported output formats."""
    LIST = 'list'
    ARRAY = 'array'
    TENSOR = 'tensor'

@dataclass
class EmbeddingConfig:
    """Configuration for embedding parameters."""
    cache_embeddings: bool = False
    batch_size: int = 32
    default_output_format: OutputFormat = OutputFormat.LIST
    max_retries: int = 3
    retry_delay: float = 1.0
    max_text_length: int = 8192