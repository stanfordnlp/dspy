from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest

from dspy.clients.openai import OpenAIProvider


@pytest.mark.parametrize(
    ("method_name", "resource_name"),
    [
        ("does_job_exist", "fine_tuning"),
        ("does_file_exist", "files"),
    ],
)
def test_provider_existence_checks_only_handle_not_found(monkeypatch, method_name, resource_name):
    retrieve = Mock()
    resource = SimpleNamespace(retrieve=retrieve)
    if resource_name == "fine_tuning":
        resource = SimpleNamespace(jobs=resource)
    monkeypatch.setitem(openai.__dict__, resource_name, resource)
    exists = getattr(OpenAIProvider, method_name)

    assert exists("resource-id") is True
    retrieve.assert_called_once_with("resource-id")

    response = httpx.Response(404, request=httpx.Request("GET", "https://api.openai.com/resource"))
    retrieve.side_effect = openai.NotFoundError("not found", response=response, body=None)
    assert exists("missing-id") is False

    response = httpx.Response(401, request=httpx.Request("GET", "https://api.openai.com/resource"))
    retrieve.side_effect = openai.AuthenticationError("unauthorized", response=response, body=None)
    with pytest.raises(openai.AuthenticationError):
        exists("private-id")

    retrieve.reset_mock()
    retrieve.side_effect = None
    assert exists(None) is False
    retrieve.assert_not_called()
