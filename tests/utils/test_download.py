from unittest.mock import MagicMock, patch

import pytest
import requests

from dspy.utils import download

ERROR_PAGE = b"<html>Service Unavailable</html>\n"


def head_response():
    # HEAD succeeds and reports a size that differs from the local file, so download() goes on to GET.
    response = MagicMock()
    response.headers = {"Content-Length": str(len(ERROR_PAGE))}
    return response


def failed_get_response(status):
    response = MagicMock()
    response.iter_content.return_value = [ERROR_PAGE]
    response.raise_for_status.side_effect = requests.HTTPError(f"{status} Server Error")
    response.__enter__.return_value = response
    return response


def test_download_http_error_keeps_existing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.jsonl").write_bytes(b'{"question": "q"}\n')

    with patch("requests.head", return_value=head_response()), \
         patch("requests.get", return_value=failed_get_response(503)):
        with pytest.raises(requests.HTTPError):
            download("https://example.com/data.jsonl")

    assert (tmp_path / "data.jsonl").read_bytes() == b'{"question": "q"}\n'


def test_download_http_error_does_not_create_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with patch("requests.head", return_value=head_response()), \
         patch("requests.get", return_value=failed_get_response(404)):
        with pytest.raises(requests.HTTPError):
            download("https://example.com/data.jsonl")

    assert not (tmp_path / "data.jsonl").exists()
