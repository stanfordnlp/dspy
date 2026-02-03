from unittest.mock import MagicMock, patch

import pytest

from dspy.retrievers.moss import MossRM


@pytest.fixture
def mock_moss_client():
    return MagicMock()


def test_moss_rm_init(mock_moss_client):
    moss_rm = MossRM(index_name="test_index", moss_client=mock_moss_client, k=5)
    assert moss_rm._index_name == "test_index"
    assert moss_rm._moss_client == mock_moss_client
    assert moss_rm.k == 5


@patch("dspy.retrievers.moss.asyncio.run")
def test_moss_rm_forward(mock_asyncio_run, mock_moss_client):
    moss_rm = MossRM(index_name="test_index", moss_client=mock_moss_client, k=2)

    # Mock the search result
    mock_doc = MagicMock()
    mock_doc.text = "test text"
    mock_doc.id = "test_id"
    mock_doc.metadata = {"key": "value"}
    mock_doc.score = 0.9

    mock_result = MagicMock()
    mock_result.docs = [mock_doc]

    mock_asyncio_run.return_value = mock_result

    results = moss_rm.forward("test query")

    assert len(results) == 1
    assert results[0].long_text == "test text"
    assert results[0].id == "test_id"
    assert results[0].score == 0.9
    mock_asyncio_run.assert_called_once()


@patch("dspy.retrievers.moss.asyncio.run")
def test_moss_rm_insert(mock_asyncio_run, mock_moss_client):
    moss_rm = MossRM(index_name="test_index", moss_client=mock_moss_client)

    moss_rm.insert({"id": "doc1", "text": "content"})

    mock_asyncio_run.assert_called_once()


@patch("dspy.retrievers.moss.asyncio.run")
def test_moss_rm_get_objects(mock_asyncio_run, mock_moss_client):
    moss_rm = MossRM(index_name="test_index", moss_client=mock_moss_client)

    mock_doc = MagicMock()
    mock_doc.id = "doc1"
    mock_doc.text = "content"
    mock_doc.metadata = {}

    mock_asyncio_run.return_value = [mock_doc]

    objects = moss_rm.get_objects(num_samples=1)

    assert len(objects) == 1
    assert objects[0]["id"] == "doc1"
    mock_asyncio_run.assert_called_once()
