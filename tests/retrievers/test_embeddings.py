import gc
import os
import tempfile
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from dspy.retrievers.embeddings import Embeddings, EmbeddingsWithScores
from dspy.utils.unbatchify import Unbatchify


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


def test_embeddings_with_scores_basic_search():
    corpus = dummy_corpus()
    retriever = EmbeddingsWithScores(corpus=corpus, embedder=dummy_embedder, k=2)

    result = retriever("A dog is barking.")

    assert result.passages == ["The dog barked at the mailman.", "The cat sat on the mat."]
    assert result.indices == [1, 0]
    assert result.scores == pytest.approx([1.0, 0.0])


def test_embeddings_with_scores_save_load():
    corpus = dummy_corpus()
    original_retriever = EmbeddingsWithScores(
        corpus=corpus,
        embedder=dummy_embedder,
        k=2,
        normalize=False,
        brute_force_threshold=1000,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_embeddings_with_scores")

        original_retriever.save(save_path)
        loaded_retriever = EmbeddingsWithScores.from_saved(save_path, dummy_embedder)

        original_result = original_retriever("cat sitting")
        loaded_result = loaded_retriever("cat sitting")

        assert loaded_result.passages == original_result.passages
        assert loaded_result.indices == original_result.indices
        assert loaded_result.scores == pytest.approx(original_result.scores)


@pytest.mark.parametrize("retriever_cls", [Embeddings, EmbeddingsWithScores])
@pytest.mark.parametrize("from_saved", [False, True])
def test_discarded_embeddings_release_corpus_and_worker(retriever_cls, from_saved, tmp_path):
    retriever = retriever_cls(dummy_corpus(), dummy_embedder, k=1)
    if from_saved:
        retriever.save(str(tmp_path))
        retriever.search_fn.close()
        retriever = retriever_cls.from_saved(str(tmp_path), dummy_embedder)

    assert retriever("dog").indices == [1]
    owner_ref = weakref.ref(retriever)
    corpus_ref = weakref.ref(retriever.corpus_embeddings)
    batcher = retriever.search_fn
    try:
        del retriever
        gc.collect()
        batcher.worker_thread.join(timeout=2)
        assert owner_ref() is None
        assert corpus_ref() is None
        assert not batcher.worker_thread.is_alive()
    finally:
        batcher.close()


def test_failed_embeddings_load_does_not_leave_worker(monkeypatch, tmp_path):
    batchers = []

    def track_batcher(*args, **kwargs):
        batcher = Unbatchify(*args, **kwargs)
        batchers.append(batcher)
        return batcher

    monkeypatch.setattr("dspy.retrievers.embeddings.Unbatchify", track_batcher)
    try:
        with pytest.raises(FileNotFoundError):
            Embeddings.from_saved(str(tmp_path / "missing"), dummy_embedder)
        assert not any(batcher.worker_thread.is_alive() for batcher in batchers)
    finally:
        for batcher in batchers:
            batcher.close()


def test_embeddings_context_manager_closes_on_error():
    retriever = Embeddings(dummy_corpus(), dummy_embedder, k=1)
    try:
        with pytest.raises(ValueError, match="caller failed"), retriever as opened:
            assert opened("dog").indices == [1]
            raise ValueError("caller failed")
        assert not retriever.search_fn.worker_thread.is_alive()
        retriever.close()
        with pytest.raises(RuntimeError, match="closed"):
            retriever("dog")
    finally:
        retriever.search_fn.close()


def test_discarded_embeddings_finish_active_search():
    started = threading.Event()
    release = threading.Event()

    def embedder(texts):
        if texts == ["dog"]:
            started.set()
            assert release.wait(timeout=2)
        return dummy_embedder(texts)

    retriever = Embeddings(dummy_corpus(), embedder, k=1)
    owner_ref = weakref.ref(retriever)
    corpus_ref = weakref.ref(retriever.corpus_embeddings)
    batcher = retriever.search_fn
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(batcher, "dog")
            try:
                assert started.wait(timeout=2)
                del retriever
                gc.collect()
                assert owner_ref() is not None
            finally:
                release.set()
            assert result.result(timeout=2)[1] == [1]
        batcher.worker_thread.join(timeout=2)
        assert not batcher.worker_thread.is_alive()
        assert owner_ref() is None
        assert corpus_ref() is None
    finally:
        release.set()
        batcher.close()
