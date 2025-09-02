from concurrent.futures import ThreadPoolExecutor

import numpy as np

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
