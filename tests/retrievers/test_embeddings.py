import numpy as np
import pytest

from dspy.retrievers.embeddings import Embeddings

@pytest.fixture
def dummy_corpus():
    return [
        "The cat sat on the mat.",
        "The dog barked at the mailman.",
        "The sun rises in the east.",
        "The quick brown fox jumps over the lazy dog.",
        "An apple a day keeps the doctor away."
    ]

@pytest.fixture
def dummy_embedder():
    def embed(texts):
        # Simple dummy embedder: convert each character's ASCII values and sum for each string
        embeddings = []
        for text in texts:
            vec = np.array([sum(ord(c) for c in text[i::10]) for i in range(10)], dtype=np.float32)
            embeddings.append(vec)
        return np.stack(embeddings)
    return embed

def test_embeddings_basic_search(dummy_corpus, dummy_embedder):
    retriever = Embeddings(
        corpus=dummy_corpus,
        embedder=dummy_embedder,
        k=2
    )

    query = "A fox is quick and brown."
    prediction = retriever(query)

    assert hasattr(prediction, 'passages')
    assert hasattr(prediction, 'indices')

    assert isinstance(prediction.passages, list)
    assert isinstance(prediction.indices, list)

    assert len(prediction.passages) == 2
    assert len(prediction.indices) == 2

    for passage in prediction.passages:
        assert isinstance(passage, str)
        assert passage in dummy_corpus

def test_embeddings_forward_batch(dummy_corpus, dummy_embedder):
    retriever = Embeddings(
        corpus=dummy_corpus,
        embedder=dummy_embedder,
        k=3
    )

    queries = ["Healthy habits", "Animals on the move"]
    batch_results = retriever._batch_forward(queries)

    assert isinstance(batch_results, list)
    assert len(batch_results) == len(queries)

    for passages, indices in batch_results:
        assert isinstance(passages, list)
        assert isinstance(indices, list)
        assert len(passages) == 3
        assert len(indices) == 3
        for p in passages:
            assert isinstance(p, str)
            assert p in dummy_corpus

def test_normalization(dummy_embedder):
    dummy_data = ["text one", "text two"]
    embeds = dummy_embedder(dummy_data)
    
    retriever = Embeddings(
        corpus=dummy_data,
        embedder=dummy_embedder,
        k=1
    )

    normalized = retriever._normalize(embeds)
    
    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)
