import pytest

from dspy.retrieve.chromadb_rm import ChromadbRM
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction, DefaultEmbeddingFunction

@pytest.mark.parametrize(
    "model_name",
    [
        (None),
        ("sentence-transformers/all-mpnet-base-v2")

    ]
)
def test_embedding_function(model_name, tmpdir):
    """Check that only one embedding function is used in the pipeline."""
    if model_name:
        collection_name = f"test-{model_name.replace('/', '_')}"
        embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
    else:
        collection_name = "test-default"
        embedding_function = DefaultEmbeddingFunction()

    rm = ChromadbRM(
        collection_name=collection_name,
        persist_directory=f"{str(tmpdir)}/.chroma",
        embedding_function=embedding_function
    )

    assert rm.ef == rm._chromadb_collection._embedding_function