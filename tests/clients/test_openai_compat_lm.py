"""Offline unit tests for the `_OpenAICompatLM` engine."""

import json
import logging
from collections import deque

import pytest
import requests

import dspy
from dspy.clients.base_lm import BaseLM
from dspy.clients.cache import Cache
from dspy.clients.openai_compat_lm import _normalize_error, _OpenAICompatLM, _RawResponse
from dspy.utils.exceptions import (
    ContextWindowExceededError,
    LMAuthError,
    LMBillingError,
    LMInvalidRequestError,
    LMProviderError,
    LMRateLimitError,
    LMServerError,
    LMTimeoutError,
    LMTransportError,
    LMUnsupportedModelError,
)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROVIDER = "openai_compat"


def _completion(text="hello", *, model=MODEL):
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def _raw(data, *, status=200, headers=None):
    body = json.dumps(data).encode() if not isinstance(data, bytes) else data
    return _RawResponse(status=status, headers=headers or {}, body=body)


class _FakePost:
    def __init__(self, *results):
        self.results = deque(results or [_raw(_completion())])
        self.calls = []

    def __call__(self, url, body, headers, timeout):
        self.calls.append({"url": url, "body": body, "headers": headers, "timeout": timeout})
        result = self.results.popleft()
        if isinstance(result, Exception):
            raise result
        return result


def _make_lm(*, post=None, **kwargs):
    defaults = {
        "model": MODEL,
        "base_url": "http://localhost:8000/v1",
        "cache": False,
        "num_retries": 0,
        "_post": post or _FakePost(),
    }
    defaults.update(kwargs)
    return _OpenAICompatLM(**defaults)


def _envelope(code=None, message="boom", type_=None):
    error = {"message": message}
    if code is not None:
        error["code"] = code
    if type_ is not None:
        error["type"] = type_
    return json.dumps({"error": error})


@pytest.fixture
def isolated_cache(tmp_path):
    original = dspy.cache
    dspy.cache = Cache(
        enable_disk_cache=False,
        enable_memory_cache=True,
        disk_cache_dir=tmp_path,
        memory_max_entries=100,
    )
    yield dspy.cache
    dspy.cache = original


class TestErrorNormalization:
    @pytest.mark.parametrize(
        "status,body,expected_cls,expected_code",
        [
            (
                400,
                _envelope("context_length_exceeded", "too long"),
                ContextWindowExceededError,
                "context_length_exceeded",
            ),
            (400, _envelope("model_not_found", "missing"), LMUnsupportedModelError, "model_not_found"),
            (400, _envelope("model_not_available", "missing"), LMUnsupportedModelError, "model_not_available"),
            (400, _envelope("unsupported_model", "missing"), LMUnsupportedModelError, "unsupported_model"),
            (402, _envelope("insufficient_quota", "quota"), LMBillingError, "insufficient_quota"),
            (401, _envelope("invalid_api_key", "bad key"), LMAuthError, "invalid_api_key"),
            (401, _envelope(type_="authentication_error"), LMAuthError, "authentication_error"),
            (429, _envelope("rate_limit_exceeded"), LMRateLimitError, "rate_limit_exceeded"),
            (429, _envelope(type_="rate_limit_error"), LMRateLimitError, "rate_limit_error"),
            (404, _envelope(message="model foo not found"), LMUnsupportedModelError, None),
            (401, _envelope(message="unauthorized"), LMAuthError, None),
            (403, _envelope(message="forbidden"), LMAuthError, None),
            (402, _envelope(message="payment required"), LMBillingError, None),
            (408, _envelope(message="timeout"), LMTimeoutError, None),
            (504, _envelope(message="timeout"), LMTimeoutError, None),
            (429, _envelope(message="slow down"), LMRateLimitError, None),
            (400, _envelope(message="bad request"), LMInvalidRequestError, None),
            (418, _envelope(message="teapot"), LMInvalidRequestError, None),
            (422, _envelope(message="unprocessable"), LMInvalidRequestError, None),
            (500, _envelope(message="failed"), LMServerError, None),
            (503, _envelope(message="failed"), LMServerError, None),
            (500, "<html>nope</html>", LMServerError, None),
            (500, "", LMServerError, None),
        ],
    )
    def test_normalize_error_maps_exact_error_class(self, status, body, expected_cls, expected_code):
        error = _normalize_error(status, body, model=MODEL, provider=PROVIDER)

        assert type(error) is expected_cls
        assert error.model == MODEL
        assert error.provider == PROVIDER
        assert error.status == status
        assert error.provider_code == expected_code

    def test_normalize_error_uses_type_when_custom_code_is_unknown(self):
        error = _normalize_error(400, _envelope("gateway_code", type_="authentication_error"))

        assert type(error) is LMAuthError
        assert error.provider_code == "gateway_code"

    @pytest.mark.parametrize("message", [123, ["bad"], {"detail": "bad"}])
    def test_normalize_error_coerces_non_string_messages(self, message):
        error = _normalize_error(404, _envelope(message=message))

        assert isinstance(error, LMInvalidRequestError)
        assert error.message == str(message)

    def test_normalize_error_extracts_response_headers(self):
        error = _normalize_error(
            429,
            _envelope(message="slow"),
            headers={"X-Request-ID": "req-1", "Retry-After": "2.5"},
        )

        assert error.request_id == "req-1"
        assert error.retry_after == 2.5


class TestRequestResponseTranslation:
    def test_forward_translates_request_and_response(self):
        post = _FakePost(_raw(_completion("answer")))
        lm = _make_lm(post=post, api_key="secret", temperature=0.2, max_tokens=64, timeout=12)
        request = dspy.LMRequest.from_call(model=MODEL, prompt="question", temperature=0.3, max_tokens=32)

        response = lm.forward(request)

        assert response.text == "answer"
        assert response.usage.total_tokens == 4
        assert post.calls == [
            {
                "url": "http://localhost:8000/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "question"}],
                    "temperature": 0.3,
                    "max_tokens": 32,
                },
                "headers": {
                    "Content-Type": "application/json",
                    "User-Agent": f"DSPy/{dspy.__version__}",
                    "Authorization": "Bearer secret",
                },
                "timeout": 12.0,
            }
        ]

    def test_multiple_choices_map_to_multiple_outputs(self):
        completion = _completion("first")
        completion["choices"].append(
            {"index": 1, "message": {"role": "assistant", "content": "second"}, "finish_reason": "stop"}
        )
        lm = _make_lm(post=_FakePost(_raw(completion)))

        response = lm.forward(dspy.LMRequest.from_call(model=MODEL, prompt="hi", n=2))

        assert [output.text for output in response.outputs] == ["first", "second"]

    def test_tool_call_round_trip(self):
        completion = {
            "id": "chatcmpl-test",
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        post = _FakePost(_raw(completion))
        lm = _make_lm(post=post, supports_function_calling=True)
        tool = dspy.core.types.LMToolSpec(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        request = dspy.LMRequest.from_call(model=MODEL, prompt="Weather in Paris?", tools=[tool])

        response = lm.forward(request)

        sent = post.calls[0]["body"]
        assert sent["tools"][0]["function"]["name"] == "get_weather"
        call = response.tool_calls[0]
        assert (call.id, call.name, call.args) == ("call_1", "get_weather", {"city": "Paris"})

    def test_vllm_reasoning_key_maps_to_thinking_part(self):
        completion = _completion(None)
        completion["choices"][0]["message"]["content"] = None
        completion["choices"][0]["message"]["reasoning"] = "thinking hard"
        lm = _make_lm(post=_FakePost(_raw(completion)))

        response = lm.forward(dspy.LMRequest.from_call(model=MODEL, prompt="hi"))

        assert response.reasoning_content == "thinking hard"

    def test_reasoning_only_truncated_output_keeps_legacy_contract(self):
        # vLLM reasoning models can burn the whole token budget thinking:
        # content=None with only reasoning. Legacy outputs must stay
        # `str | dict | None` so adapter re-normalization does not crash.
        completion = _completion(None)
        completion["choices"][0]["message"]["content"] = None
        completion["choices"][0]["message"]["reasoning"] = "partial thought"
        completion["choices"][0]["finish_reason"] = "length"
        lm = _make_lm(post=_FakePost(_raw(completion)))

        outputs = lm("hi")

        assert outputs == [{"text": None, "reasoning_content": "partial thought"}]

    def test_empty_output_to_value_is_none(self):
        from dspy.core.types import LMOutput

        assert LMOutput(parts=[]).to_value() is None

    def test_truncated_response_logs_warning(self, caplog, monkeypatch):
        completion = _completion("cut off")
        completion["choices"][0]["finish_reason"] = "length"
        lm = _make_lm(post=_FakePost(_raw(completion)))

        # The dspy logger does not propagate to the root logger by default.
        monkeypatch.setattr(logging.getLogger("dspy"), "propagate", True)
        with caplog.at_level("WARNING", logger="dspy.clients.openai_compat_lm"):
            lm.forward(dspy.LMRequest.from_call(model=MODEL, prompt="hi", max_tokens=8))

        assert "truncated" in caplog.text
        assert "max_tokens=8" in caplog.text

    def test_extra_headers_explicitly_override_defaults(self):
        post = _FakePost()
        lm = _make_lm(post=post, api_key="secret", extra_headers={"Authorization": "Token custom", "X-Route": "blue"})

        lm.forward(dspy.LMRequest.from_call(model=MODEL, prompt="hi"))

        assert post.calls[0]["headers"]["Authorization"] == "Token custom"
        assert post.calls[0]["headers"]["X-Route"] == "blue"


class TestTransportErrors:
    @pytest.mark.parametrize(
        "exception,expected_cls",
        [
            (requests.Timeout("late"), LMTimeoutError),
            (requests.ConnectionError("down"), LMTransportError),
            (TimeoutError("late"), LMTimeoutError),
            (OSError("down"), LMTransportError),
        ],
    )
    def test_transport_exceptions_are_normalized(self, exception, expected_cls):
        lm = _make_lm(post=_FakePost(exception))

        with pytest.raises(expected_cls) as exc_info:
            lm.forward(dspy.LMRequest.from_call(model=MODEL, prompt="hi"))

        assert exc_info.value.__cause__ is exception

    def test_http_errors_are_raised_from_forward(self):
        lm = _make_lm(post=_FakePost(_raw({"error": {"message": "slow"}}, status=429)))

        with pytest.raises(LMRateLimitError):
            lm.forward(dspy.LMRequest.from_call(model=MODEL, prompt="hi"))

    @pytest.mark.parametrize("body", [b"not json", b"[]", b"{}"])
    def test_invalid_success_responses_raise_provider_error(self, body):
        lm = _make_lm(post=_FakePost(_raw(body)))

        with pytest.raises(LMProviderError):
            lm.forward(dspy.LMRequest.from_call(model=MODEL, prompt="hi"))


class TestRetries:
    def test_retryable_error_retries_using_retry_after(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("dspy.clients.openai_compat_lm.time.sleep", sleeps.append)
        post = _FakePost(
            _raw({"error": {"message": "slow"}}, status=429, headers={"retry-after": "0.25"}),
            _raw(_completion("ok")),
        )
        lm = _make_lm(post=post, num_retries=2)

        assert lm.forward(dspy.LMRequest.from_call(model=MODEL, prompt="hi")).text == "ok"
        assert len(post.calls) == 2
        assert sleeps == [0.25]

    def test_retry_backoff_is_exponential_without_retry_after(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("dspy.clients.openai_compat_lm.time.sleep", sleeps.append)
        post = _FakePost(
            _raw({"error": {"message": "slow"}}, status=429),
            _raw({"error": {"message": "slow"}}, status=429),
            _raw(_completion("ok")),
        )
        lm = _make_lm(post=post, num_retries=2)

        assert lm.forward(dspy.LMRequest.from_call(model=MODEL, prompt="hi")).text == "ok"
        assert sleeps == [1, 2]

    def test_non_retryable_error_is_not_retried(self):
        post = _FakePost(_raw({"error": {"message": "bad"}}, status=400), _raw(_completion()))
        lm = _make_lm(post=post, num_retries=2)

        with pytest.raises(LMInvalidRequestError):
            lm.forward(dspy.LMRequest.from_call(model=MODEL, prompt="hi"))
        assert len(post.calls) == 1


class TestCaching:
    def test_cache_avoids_second_post_and_cache_false_overrides(self, isolated_cache):
        post = _FakePost(_raw(_completion("cached")), _raw(_completion("fresh")), _raw(_completion("again")))
        lm = _make_lm(post=post, cache=True)

        first = lm("same")
        second = lm("same")
        uncached = lm("same", cache=False)

        assert first == ["cached"]
        assert second == ["cached"]
        assert uncached == ["fresh"]
        assert len(post.calls) == 2
        assert lm.history[-2].response.cache_hit is True

    def test_cache_works_with_pydantic_response_format(self, isolated_cache):
        import pydantic

        class Answer(pydantic.BaseModel):
            value: str

        post = _FakePost(_raw(_completion("structured")), _raw(_completion("unexpected")))
        lm = _make_lm(post=post, cache=True)
        request = dspy.LMRequest.from_call(model=MODEL, prompt="same", response_format=Answer)

        assert lm(request).text == "structured"
        assert lm(request).text == "structured"
        assert len(post.calls) == 1

    def test_cache_key_varies_with_rollout_id(self, isolated_cache):
        post = _FakePost(_raw(_completion("r0")), _raw(_completion("r1")))
        lm = _make_lm(post=post, cache=True, temperature=1.0)

        assert lm("same", rollout_id=0) == ["r0"]
        assert lm("same", rollout_id=1) == ["r1"]
        assert lm("same", rollout_id=0) == ["r0"]
        assert len(post.calls) == 2

    def test_cache_identity_distinguishes_endpoint_and_credentials(self, isolated_cache):
        request = dspy.LMRequest.from_call(model=MODEL, prompt="same")
        first_post = _FakePost(_raw(_completion("first")))
        second_post = _FakePost(_raw(_completion("second")))
        third_post = _FakePost(_raw(_completion("third")))

        first = _make_lm(post=first_post, cache=True, api_key="one")
        second = _make_lm(post=second_post, cache=True, api_key="two")
        third = _make_lm(post=third_post, cache=True, api_key="one", base_url="http://other/v1")

        assert first(request).text == "first"
        assert second(request).text == "second"
        assert third(request).text == "third"


class TestSyncAsyncAndPublicContract:
    @pytest.mark.asyncio
    async def test_aforward_matches_forward(self):
        sync_lm = _make_lm(post=_FakePost(_raw(_completion("same"))))
        async_lm = _make_lm(post=_FakePost(_raw(_completion("same"))))
        request = dspy.LMRequest.from_call(model=MODEL, prompt="hi")

        assert (await async_lm.aforward(request)).model_dump() == sync_lm.forward(request).model_dump()

    def test_legacy_and_typed_public_returns(self):
        lm = _make_lm(post=_FakePost(_raw(_completion("legacy")), _raw(_completion("typed"))))

        assert lm("hi") == ["legacy"]
        with dspy.context(experimental=True):
            response = lm(dspy.User("hi"))
        assert isinstance(response, dspy.LMResponse)
        assert response.text == "typed"


class TestConfigurationAndCapabilities:
    @pytest.mark.parametrize(
        "kwargs,env,expected",
        [
            (
                {"api_key": "explicit", "api_key_env": "GATEWAY_KEY"},
                {"GATEWAY_KEY": "gateway", "OPENAI_API_KEY": "openai"},
                "explicit",
            ),
            ({"api_key_env": "GATEWAY_KEY"}, {"GATEWAY_KEY": "gateway", "OPENAI_API_KEY": "openai"}, "gateway"),
            ({"use_openai_api_key_env": True}, {"OPENAI_API_KEY": "openai"}, "openai"),
            ({}, {"OPENAI_API_KEY": "openai"}, None),
            ({}, {}, None),
        ],
    )
    def test_api_key_resolution(self, monkeypatch, kwargs, env, expected):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GATEWAY_KEY", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)

        assert _make_lm(**kwargs)._resolved_api_key() == expected

    def test_require_auth_fails_locally_without_credential(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        post = _FakePost()
        lm = _make_lm(post=post, require_auth=True, api_key_env="GATEWAY_KEY")

        with pytest.raises(dspy.LMNotConfiguredError, match="GATEWAY_KEY"):
            lm.forward(dspy.LMRequest.from_call(model=MODEL, prompt="hi"))
        assert post.calls == []

        monkeypatch.setenv("GATEWAY_KEY", "now-set")
        assert lm.forward(dspy.LMRequest.from_call(model=MODEL, prompt="hi")).text == "hello"

    def test_require_auth_round_trips_through_state(self):
        state = _make_lm(require_auth=True, api_key="k").dump_state()

        assert state["engine"]["require_auth"] is True
        assert BaseLM.load_state(state).require_auth is True

    def test_callable_api_key_is_resolved_per_request(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        tokens = iter(["token-1", "token-2"])
        post = _FakePost(_raw(_completion()), _raw(_completion()))
        lm = _make_lm(post=post, api_key=lambda: next(tokens))

        lm.forward(dspy.LMRequest.from_call(model=MODEL, prompt="hi"))
        lm.forward(dspy.LMRequest.from_call(model=MODEL, prompt="hi"))

        sent = [call["headers"]["Authorization"] for call in post.calls]
        assert sent == ["Bearer token-1", "Bearer token-2"]

    def test_callable_api_key_is_never_serialized(self):
        lm = _make_lm(api_key=lambda: "secret-token")

        state = lm.dump_state()

        assert "secret-token" not in json.dumps(state)
        assert state["engine"]["use_openai_api_key_env"] is False

    @pytest.mark.parametrize("flag", ["supports_function_calling", "supports_reasoning", "supports_response_schema"])
    def test_capabilities_default_false_and_honor_opt_in(self, flag):
        assert getattr(_make_lm(), flag) is False
        assert getattr(_make_lm(**{flag: True}), flag) is True

    def test_supported_params_follow_capabilities(self):
        lm = _make_lm(supports_function_calling=True, supports_reasoning=True, supports_response_schema=True)

        assert {
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning_effort",
            "response_format",
        } <= lm.supported_params

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"model_type": "text"}, "model_type='chat'"),
            ({"base_url": ""}, "non-empty base_url"),
            ({"timeout": 0}, "greater than zero"),
            ({"num_retries": -1}, "cannot be negative"),
        ],
    )
    def test_invalid_configuration_raises_structured_error(self, kwargs, match):
        with pytest.raises(dspy.LMConfigurationError, match=match):
            _make_lm(**kwargs)


class TestSerializationAndExports:
    def test_dump_state_is_router_state_and_loads_back_as_engine(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GATEWAY_KEY", raising=False)
        lm = _make_lm(
            api_key="explicit-secret",
            api_key_env="GATEWAY_KEY",
            extra_headers={"X-Custom": "yes", "Authorization": "other-secret", "X-API-Key": "secret"},
            timeout=42,
            supports_function_calling=True,
            temperature=0.3,
            max_tokens=256,
        )

        state = lm.dump_state()
        serialized = json.dumps(state)

        assert "explicit-secret" not in serialized
        assert "other-secret" not in serialized
        assert '"X-API-Key"' not in serialized
        assert "OpenAICompatLM" not in serialized
        assert state["model"] == f"openai/{MODEL}"
        assert state["api_base"] == lm.base_url

        engine = state["engine"]
        assert engine["engine"] == "openai_compat"
        assert engine["model"] == MODEL
        assert engine["base_url"] == lm.base_url
        assert engine["api_key_env"] == "GATEWAY_KEY"
        assert engine["use_openai_api_key_env"] is False
        assert engine["supports_function_calling"] is True
        assert engine["extra_headers"] == {"X-Custom": "yes"}

        loaded = BaseLM.load_state(state)

        assert type(loaded) is _OpenAICompatLM
        assert loaded.model == MODEL
        assert loaded.base_url == lm.base_url
        assert loaded.api_key_env == "GATEWAY_KEY"
        assert loaded.timeout == 42.0
        assert loaded.supports_function_calling is True
        assert loaded.extra_headers == {"X-Custom": "yes"}
        assert loaded.kwargs["temperature"] == 0.3
        assert loaded.kwargs["max_tokens"] == 256
        assert loaded._resolved_api_key() is None
        assert loaded.use_openai_api_key_env is False

    def test_engine_state_survives_a_second_save_cycle(self):
        state = _make_lm(require_auth=True).dump_state()
        loaded = BaseLM.load_state(state)

        assert loaded.dump_state()["engine"] == state["engine"]

    def test_unknown_engine_name_falls_back_to_lm_and_carries_block(self):
        from dspy.clients.lm import LM

        state = _make_lm().dump_state()
        state["engine"]["engine"] = "engine_from_the_future"

        loaded = BaseLM.load_state(state)

        assert type(loaded) is LM
        assert loaded.model == f"openai/{MODEL}"
        assert loaded.kwargs["api_base"] == "http://localhost:8000/v1"
        assert loaded._engine_state["engine"] == "engine_from_the_future"
        assert loaded.dump_state()["engine"] == state["engine"]

    def test_saved_predict_program_load_paths(self, tmp_path, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        predict = dspy.Predict("q -> a")
        predict.lm = _make_lm(api_key="secret", supports_function_calling=True)
        path = tmp_path / "program.json"
        predict.save(str(path))

        untrusted = dspy.Predict("q -> a")
        with pytest.raises(dspy.LMStateError, match="engine"):
            untrusted.load(str(path))

        replacement = dspy.LM(model="openai/gpt-4o-mini")
        with_lm = dspy.Predict("q -> a")
        with_lm.load(str(path), lm=replacement)
        assert with_lm.lm is replacement

        trusted = dspy.Predict("q -> a")
        trusted.load(str(path), allow_unsafe_lm_state=True)
        assert type(trusted.lm) is _OpenAICompatLM
        assert trusted.lm.base_url == "http://localhost:8000/v1"
        assert trusted.lm.supports_function_calling is True
        assert trusted.lm._resolved_api_key() is None

    def test_openai_compat_lm_is_not_exported_at_top_level(self):
        assert not hasattr(dspy, "OpenAICompatLM")

    @pytest.mark.parametrize("api_key", ["sk-live-SECRET", lambda: "sk-live-SECRET"])
    def test_program_pickle_scrubs_engine_secrets_and_history(self, tmp_path, api_key):
        lm = _OpenAICompatLM(
            MODEL,
            "http://localhost:8000/v1",
            api_key=api_key,
            cache=False,
            extra_headers={"Authorization": "Token header-SECRET", "X-Route": "blue"},
        )
        lm.history.append({"prompt": "PRIVATE-PROMPT"})
        predict = dspy.Predict("q -> a")
        predict.lm = lm
        predict.save(tmp_path / "program", save_program=True)

        raw = (tmp_path / "program" / "program.pkl").read_bytes()
        assert b"SECRET" not in raw
        assert b"PRIVATE-PROMPT" not in raw

        loaded = dspy.load(str(tmp_path / "program"), allow_pickle=True)
        assert loaded.lm._resolved_api_key() is None
        assert loaded.lm.extra_headers == {"X-Route": "blue"}
        assert loaded.lm.history == []
        assert loaded.lm._post is not None

    def test_copy_and_deepcopy_keep_credentials_and_history(self):
        import copy

        lm = _make_lm(api_key="sk-keep")
        lm.history.append("entry")

        deep = copy.deepcopy(lm)
        assert deep._resolved_api_key() == "sk-keep"
        assert deep.history == ["entry"]

        shallow = lm.copy()
        assert shallow._resolved_api_key() == "sk-keep"
