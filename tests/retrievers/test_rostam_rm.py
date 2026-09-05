from unittest.mock import MagicMock

import pytest

pytest.importorskip("rostam")

from dspy.retrievers.rostam_rm import RostamRM


class _FakeDoc:
    def __init__(self, content):
        self.content = content


def _fake_embedder(text):
    return [float(len(text)), 0.0, 0.0]


def _make_retriever(**kwargs):
    client = MagicMock()
    retriever = RostamRM(
        "my_collection",
        rostam_client=client,
        embedder=_fake_embedder,
        **kwargs,
    )
    return retriever, client


def test_requires_exactly_one_of_client_or_url():
    with pytest.raises(ValueError):
        RostamRM("my_collection", embedder=_fake_embedder)

    with pytest.raises(ValueError):
        RostamRM(
            "my_collection",
            rostam_client=MagicMock(),
            url="http://localhost:8080",
            embedder=_fake_embedder,
        )


def test_forward_embeds_query_and_returns_passages():
    retriever, client = _make_retriever(k=2)
    client.search_docs.return_value = [_FakeDoc("doc one"), _FakeDoc("doc two")]

    result = retriever("what is rostam")

    client.search_docs.assert_called_once_with("my_collection", _fake_embedder("what is rostam"), 2, filter=None)
    # forward returns the retrieval-model contract dspy.Retrieve consumes: a list
    # of dotdicts exposing long_text (NOT a Prediction — that broke the configured
    # rm path, which does [psg.long_text for psg in rm_result]).
    assert [p.long_text for p in result] == ["doc one", "doc two"]


def test_forward_k_and_filter_override_defaults():
    retriever, client = _make_retriever(k=3, filter={"tag": "default"})
    client.search_docs.return_value = []

    retriever("query", k=1, filter={"tag": "override"})

    client.search_docs.assert_called_once_with("my_collection", _fake_embedder("query"), 1, filter={"tag": "override"})


def test_forward_accepts_multiple_queries():
    retriever, client = _make_retriever(k=1)
    client.search_docs.side_effect = [[_FakeDoc("a")], [_FakeDoc("b")]]

    result = retriever(["q1", "q2"])

    assert [p.long_text for p in result] == ["a", "b"]
    assert client.search_docs.call_count == 2


def test_works_as_configured_retrieval_model():
    """Regression: as the global rm, dspy.Retrieve reads .long_text off each item
    forward returns. RostamRM used to return a Prediction, which dspy.Retrieve
    iterated as its field names -> AttributeError. It must return long_text items."""
    import dspy

    retriever, client = _make_retriever(k=2)
    client.search_docs.return_value = [_FakeDoc("first"), _FakeDoc("second")]
    with dspy.context(rm=retriever):
        passages = dspy.Retrieve(k=2)("a query").passages
    assert passages == ["first", "second"]


def test_index_validates_lengths_before_mutating():
    """Regression: a length mismatch must be rejected BEFORE any server mutation —
    previously index() created the collection and upserted the first doc, then
    raised, leaving the store partially written by a call that reports failure."""
    retriever, client = _make_retriever()

    with pytest.raises(ValueError):
        retriever.index(["a", "b"], embeddings=[[0.0, 0.0, 0.0]])  # 2 texts, 1 embedding

    client.create_collection.assert_not_called()
    client.upsert.assert_not_called()


def test_index_creates_collection_once_and_upserts():
    retriever, client = _make_retriever()

    ids = retriever.index(["hello", "world"])

    client.create_collection.assert_called_once_with("my_collection", 3, metric="cosine")
    assert client.upsert.call_count == 2
    assert len(ids) == 2

    # A second index() call must not attempt to recreate the collection.
    retriever.index(["again"])
    client.create_collection.assert_called_once()


def test_index_with_explicit_ids_and_metadata():
    retriever, client = _make_retriever(auto_create=False)

    ids = retriever.index(
        ["hello"],
        ids=["42"],
        metadatas=[{"source": "unit-test", "tags": ["a", "b"], "bad": {"nested": True}}],
    )

    client.create_collection.assert_not_called()
    assert ids == ["42"]
    client.upsert.assert_called_once_with(
        "my_collection",
        42,
        _fake_embedder("hello"),
        content="hello",
        metadata={"source": "unit-test", "tags": ["a", "b"]},
    )


def test_index_empty_list_is_a_no_op():
    retriever, client = _make_retriever()

    assert retriever.index([]) == []
    client.upsert.assert_not_called()
    client.create_collection.assert_not_called()
