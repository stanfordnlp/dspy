import pytest
from dspy.embeddings.config import EmbeddingConfig, OutputFormat


def test_embedding_config_defaults():
    config = EmbeddingConfig()
    assert config.cache_embeddings == False
    assert config.batch_size == 32
    assert config.default_output_format == OutputFormat.LIST
    assert config.max_retries == 3
    assert config.retry_delay == 1.0
    assert config.max_text_length == 8192


def test_embedding_config_custom():
    config = EmbeddingConfig(
        cache_embeddings=True,
        batch_size=64,
        default_output_format=OutputFormat.ARRAY,
        max_retries=5,
        retry_delay=2.0,
        max_text_length=10000
    )
    assert config.cache_embeddings == True
    assert config.batch_size == 64
    assert config.default_output_format == OutputFormat.ARRAY
    assert config.max_retries == 5
    assert config.retry_delay == 2.0
    assert config.max_text_length == 10000


def test_output_format_enum():
    assert OutputFormat.LIST == 'list'
    assert OutputFormat.ARRAY == 'array'
    assert OutputFormat.TENSOR == 'tensor'
