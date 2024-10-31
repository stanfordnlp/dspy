import pytest
from dspy.embeddings.providers.utils import generate_cache_key, validate_api_response


def test_generate_cache_key():
    text = "hello world"
    key1 = generate_cache_key(text)
    key2 = generate_cache_key(text)
    assert key1 == key2

    text2 = "another text"
    key3 = generate_cache_key(text2)
    assert key1 != key3


def test_validate_api_response():
    response = {
        'data': [
            {'embedding': [0.1, 0.2, 0.3]},
            {'embedding': [0.4, 0.5, 0.6]},
        ]
    }
    embeddings = validate_api_response(response)
    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2, 0.3]


def test_validate_api_response_missing_data():
    response = {}
    with pytest.raises(ValueError, match="Invalid API response format"):
        validate_api_response(response)


def test_validate_api_response_missing_embedding():
    response = {
        'data': [
            {'embedding': [0.1, 0.2, 0.3]},
            {'embedding': None},
        ]
    }
    with pytest.raises(ValueError, match="Missing embeddings in response"):
        validate_api_response(response)


def test_validate_api_response_wrong_dimensions():
    response = {
        'data': [
            {'embedding': [0.1, 0.2, 0.3]},
            {'embedding': [0.4, 0.5]},
        ]
    }
    with pytest.raises(ValueError, match="Incorrect embedding dimensions"):
        validate_api_response(response, dimensions=3)
