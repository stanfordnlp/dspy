import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from dspy.retrievers.embeddings import Embeddings


def dummy_corpus():
    return [
        "The cat sat on the mat.",
        "The dog barked at the mailman.",
        "Birds fly in the sky.",
    ]


def dummy_embedder(texts):
    embeddings = []
    for text in texts:
        if "cat" in text:
            embeddings.append(np.array([1, 0, 0], dtype=np.float32))
        elif "dog" in text:
            embeddings.append(np.array([0, 1, 0], dtype=np.float32))
        else:
            embeddings.append(np.array([0, 0, 1], dtype=np.float32))
    return np.stack(embeddings)


def test_embeddings_basic_search():
    corpus = dummy_corpus()
    embedder = dummy_embedder

    retriever = Embeddings(corpus=corpus, embedder=embedder, k=1)

    query = "I saw a dog running."
    result = retriever(query)

    assert hasattr(result, "passages")
    assert hasattr(result, "indices")

    assert isinstance(result.passages, list)
    assert isinstance(result.indices, list)

    assert len(result.passages) == 1
    assert len(result.indices) == 1

    assert result.passages[0] == "The dog barked at the mailman."


def test_embeddings_multithreaded_search():
    corpus = dummy_corpus()
    embedder = dummy_embedder

    retriever = Embeddings(corpus=corpus, embedder=embedder, k=1)

    queries = [
        ("A cat is sitting on the mat.", "The cat sat on the mat."),
        ("My dog is awesome!", "The dog barked at the mailman."),
        ("Birds flying high.", "Birds fly in the sky."),
    ] * 10

    def worker(query_text, expected_passage):
        result = retriever(query_text)
        assert result.passages[0] == expected_passage
        return result.passages[0]

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker, q, expected) for q, expected in queries]
        # Results will be in original order
        results = [f.result() for f in futures]
        assert results[0] == "The cat sat on the mat."
        assert results[1] == "The dog barked at the mailman."
        assert results[2] == "Birds fly in the sky."


def test_embeddings_save_load():
    corpus = dummy_corpus()
    embedder = dummy_embedder

    original_retriever = Embeddings(corpus=corpus, embedder=embedder, k=2, normalize=False, brute_force_threshold=1000)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_embeddings")

        # Save original
        original_retriever.save(save_path)

        # Verify files were created
        assert os.path.exists(os.path.join(save_path, "config.json"))
        assert os.path.exists(os.path.join(save_path, "corpus_embeddings.npy"))
        assert not os.path.exists(os.path.join(save_path, "faiss_index.bin"))  # No FAISS for small corpus

        # Load into new instance
        new_retriever = Embeddings(corpus=["dummy"], embedder=embedder, k=1, normalize=True, brute_force_threshold=500)
        new_retriever.load(save_path, embedder)

        # Verify configuration was loaded correctly
        assert new_retriever.corpus == corpus
        assert new_retriever.k == 2
        assert new_retriever.normalize is False
        assert new_retriever.embedder == embedder
        assert new_retriever.index is None

        # Verify search results are preserved
        query = "cat sitting"
        original_result = original_retriever(query)
        loaded_result = new_retriever(query)
        assert loaded_result.passages == original_result.passages
        assert loaded_result.indices == original_result.indices


def test_embeddings_from_saved():
    corpus = dummy_corpus()
    embedder = dummy_embedder

    original_retriever = Embeddings(corpus=corpus, embedder=embedder, k=3, normalize=True, brute_force_threshold=1000)

    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_embeddings")

        original_retriever.save(save_path)
        loaded_retriever = Embeddings.from_saved(save_path, embedder)

        assert loaded_retriever.k == original_retriever.k
        assert loaded_retriever.normalize == original_retriever.normalize
        assert loaded_retriever.corpus == original_retriever.corpus



def test_embeddings_load_nonexistent_path():
    with pytest.raises((FileNotFoundError, OSError)):
        Embeddings.from_saved("/nonexistent/path", dummy_embedder)
