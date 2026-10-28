"""Contracts of the offline server used by LM and streaming integration tests."""

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import dspy
from tests.test_utils.server import read_test_server_request_logs


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_server_completion_usage_and_cache(openai_test_server, asynchronous):
    url, log_path = openai_test_server
    lm = dspy.LM("openai/dspy-test-model", api_base=url, api_key="test", cache=True)
    for _ in range(2):
        result = await lm.acall("hello") if asynchronous else lm("hello")
        assert result == ["Hi!"]
    assert lm.history[0]["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 10,
        "total_tokens": 20,
        "prompt_tokens_details": None,
        "completion_tokens_details": None,
    }
    assert lm.history[1]["usage"] == {}
    requests = read_test_server_request_logs(log_path)
    assert len(requests) == 1
    assert requests[0]["model"] == "dspy-test-model"
    assert requests[0]["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize("status", [400, 401, 429, 504])
def test_server_errors_are_logged(openai_test_server, status):
    url, log_path = openai_test_server
    body = {"model": "dspy-test-model", "messages": [{"role": "user", "content": str(status)}]}
    request = Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with pytest.raises(HTTPError) as exc:
        urlopen(request)
    assert exc.value.code == status
    assert json.load(exc.value)["error"]["code"] == str(status)
    exc.value.close()
    assert read_test_server_request_logs(log_path) == [body]


@pytest.mark.asyncio
async def test_server_streaming_prediction_and_cache(openai_test_server):
    url, log_path = openai_test_server
    lm = dspy.LM("openai/dspy-test-model", api_base=url, api_key="test", cache=True)
    program = dspy.streamify(dspy.Predict("input_text -> output_text"))
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
        for _ in range(2):
            values = [value async for value in program(input_text="hello")]
            assert values[-1].output_text == "Hello!"
    assert len(read_test_server_request_logs(log_path)) == 1
    assert lm.history[0]["usage"]["total_tokens"] == 0
    assert lm.history[1]["usage"] == {}
