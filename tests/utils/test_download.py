from unittest.mock import MagicMock, patch

import pytest
import requests

from dspy.utils import download


def error_response(status):
    response = MagicMock()
    response.headers = {"Content-Length": "34"}
    response.iter_content.return_value = [b"<html>Service Unavailable</html>\n"]
    response.raise_for_status.side_effect = requests.HTTPError(f"{status} Server Error")
    response.__enter__.return_value = response
    return response


def test_download_http_error_keeps_existing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.jsonl").write_bytes(b'{"question": "q"}\n')

    response = error_response(503)
    with patch("requests.head", return_value=response), patch("requests.get", return_value=response):
        with pytest.raises(requests.HTTPError):
            download("https://example.com/data.jsonl")

    assert (tmp_path / "data.jsonl").read_bytes() == b'{"question": "q"}\n'


def test_download_http_error_does_not_create_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    response = error_response(404)
    with patch("requests.head", return_value=response), patch("requests.get", return_value=response):
        with pytest.raises(requests.HTTPError):
            download("https://example.com/data.jsonl")

    assert not (tmp_path / "data.jsonl").exists()
