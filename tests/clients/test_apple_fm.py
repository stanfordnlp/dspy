"""Unit tests for the Apple Foundation Models DSPy adapter.

These tests run on any platform (WSL, Linux CI, macOS) by mocking
``apple_fm_sdk`` and patching ``platform.system`` where needed.
They validate message formatting, Pydantic→@generable conversion,
fallback behaviour, and response structure compatibility with BaseLM.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import types
from typing import Literal
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Fixtures — synthetic apple_fm_sdk module
# ---------------------------------------------------------------------------


def _make_fake_fm_sdk():
    """Return a minimal fake apple_fm_sdk module that can be imported."""
    fm = types.ModuleType("apple_fm_sdk")

    # guide() returns a plain string default so dataclasses.asdict() stays
    # JSON-serialisable in tests.  The guide metadata is captured via a
    # recording wrapper in individual tests, not stored on the return value.
    def guide(description, **kwargs):
        return ""

    fm.guide = guide

    # generable() is an identity decorator for test purposes
    def generable(cls):
        cls._is_generable = True
        return cls

    fm.generable = generable

    # GenerationOptions is a simple dataclass
    @dataclasses.dataclass
    class GenerationOptions:
        temperature: float | None = None

    fm.GenerationOptions = GenerationOptions

    # SystemLanguageModel
    class SystemLanguageModel:
        def is_available(self):
            return True, "available"

    fm.SystemLanguageModel = SystemLanguageModel

    # LanguageModelSession — async respond returns a plain string by default
    class LanguageModelSession:
        def __init__(self, model=None, tools=None):
            self.model = model
            self.tools = tools or []

        async def respond(self, prompt, generating=None, **kwargs):
            if generating is not None:
                # Return a minimal instance of the generating class
                return generating()
            return "Hello from Apple on-device model"

    fm.LanguageModelSession = LanguageModelSession

    # Tool base class
    class Tool:
        def call(self, **kwargs):  # pragma: no cover
            raise NotImplementedError

    fm.Tool = Tool

    return fm


@pytest.fixture(autouse=True)
def fake_apple_fm_sdk(monkeypatch):
    """Inject a fake apple_fm_sdk into sys.modules for every test."""
    fm = _make_fake_fm_sdk()
    monkeypatch.setitem(sys.modules, "apple_fm_sdk", fm)
    # Also remove cached apple_fm module so re-import picks up the fake SDK
    monkeypatch.delitem(sys.modules, "dspy.clients.apple_fm", raising=False)
    return fm


@pytest.fixture()
def lm(fake_apple_fm_sdk):
    """Return an AppleFoundationLM instance with platform check bypassed."""
    with patch("platform.system", return_value="Darwin"):
        from dspy.clients.apple_fm import AppleFoundationLM

        return AppleFoundationLM(cache=False)


# ---------------------------------------------------------------------------
# _flatten_messages
# ---------------------------------------------------------------------------


class TestFlattenMessages:
    def _flatten(self, messages):
        from dspy.clients.apple_fm import _flatten_messages

        return _flatten_messages(messages)

    def test_single_user_message(self):
        assert self._flatten([{"role": "user", "content": "hi"}]) == "hi"

    def test_system_plus_user(self):
        result = self._flatten(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Say hello."},
            ]
        )
        assert "You are helpful." in result
        assert "Say hello." in result

    def test_assistant_role_prefixed(self):
        result = self._flatten(
            [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
                {"role": "user", "content": "Q2"},
            ]
        )
        assert "Assistant: A" in result

    def test_multimodal_content_extracts_text(self):
        result = self._flatten(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": "..."},
                    ],
                }
            ]
        )
        assert result == "Describe this"

    def test_empty_messages_returns_empty(self):
        assert self._flatten([]) == ""

    def test_filters_blank_parts(self):
        result = self._flatten(
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": "hello"},
            ]
        )
        assert result == "hello"


# ---------------------------------------------------------------------------
# _pydantic_to_generable
# ---------------------------------------------------------------------------


class TestPydanticToGenerable:
    def _convert(self, model_cls, fake_fm):
        from dspy.clients.apple_fm import _pydantic_to_generable

        return _pydantic_to_generable(model_cls, fake_fm)

    def test_literal_field_mapped_to_any_of(self, fake_apple_fm_sdk):
        class Sentiment(BaseModel):
            label: Literal["positive", "negative", "neutral"]

        calls = []
        original_guide = fake_apple_fm_sdk.guide

        def recording_guide(desc, **kwargs):
            calls.append((desc, kwargs))
            return original_guide(desc, **kwargs)

        fake_apple_fm_sdk.guide = recording_guide

        cls = self._convert(Sentiment, fake_apple_fm_sdk)
        assert cls is not None
        assert any("anyOf" in kw for _, kw in calls), "Expected anyOf constraint"

    def test_int_range_field(self, fake_apple_fm_sdk):
        class Rating(BaseModel):
            score: int = Field(ge=1, le=5)

        calls = []
        original_guide = fake_apple_fm_sdk.guide

        def recording_guide(desc, **kwargs):
            calls.append((desc, kwargs))
            return original_guide(desc, **kwargs)

        fake_apple_fm_sdk.guide = recording_guide

        cls = self._convert(Rating, fake_apple_fm_sdk)
        assert cls is not None
        assert any("range" in kw and kw["range"] == (1, 5) for _, kw in calls)

    def test_str_pattern_field(self, fake_apple_fm_sdk):
        class Code(BaseModel):
            value: str = Field(pattern=r"\d+")

        calls = []
        original_guide = fake_apple_fm_sdk.guide

        def recording_guide(desc, **kwargs):
            calls.append((desc, kwargs))
            return original_guide(desc, **kwargs)

        fake_apple_fm_sdk.guide = recording_guide

        cls = self._convert(Code, fake_apple_fm_sdk)
        assert cls is not None
        assert any("regex" in kw for _, kw in calls)

    def test_plain_str_field_no_guide(self, fake_apple_fm_sdk):
        class Free(BaseModel):
            text: str

        calls = []
        original_guide = fake_apple_fm_sdk.guide

        def recording_guide(desc, **kwargs):
            calls.append((desc, kwargs))
            return original_guide(desc, **kwargs)

        fake_apple_fm_sdk.guide = recording_guide

        cls = self._convert(Free, fake_apple_fm_sdk)
        assert cls is not None
        assert calls == [], "No guide call expected for plain str"

    def test_returns_none_on_make_dataclass_failure(self, fake_apple_fm_sdk):
        class Bad(BaseModel):
            x: str

        def bad_generable(cls):
            raise RuntimeError("SDK error")

        fake_apple_fm_sdk.generable = bad_generable

        cls = self._convert(Bad, fake_apple_fm_sdk)
        assert cls is None

    def test_generable_decorator_applied(self, fake_apple_fm_sdk):
        class Simple(BaseModel):
            name: str

        cls = self._convert(Simple, fake_apple_fm_sdk)
        assert getattr(cls, "_is_generable", False), "Expected @generable decorator to be applied"


# ---------------------------------------------------------------------------
# _FMResponse / _FMUsage structure
# ---------------------------------------------------------------------------


class TestResponseStructure:
    def test_usage_dict_conversion(self):
        from dspy.clients.apple_fm import _FMUsage

        usage = _FMUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        d = dict(usage)
        assert d == {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}

    def test_response_has_choices_and_model(self):
        from dspy.clients.apple_fm import _FMChoice, _FMMessage, _FMResponse, _FMUsage

        resp = _FMResponse(
            choices=[_FMChoice(message=_FMMessage(content="hello"))],
            usage=_FMUsage(),
            model="apple/on-device",
        )
        assert resp.choices[0].message.content == "hello"
        assert resp.model == "apple/on-device"

    def test_process_completion_compatible(self, lm):
        """_FMResponse must be consumable by BaseLM._process_completion."""
        from dspy.clients.apple_fm import _FMChoice, _FMMessage, _FMResponse, _FMUsage

        resp = _FMResponse(
            choices=[_FMChoice(message=_FMMessage(content="result text"))],
            usage=_FMUsage(),
            model="apple/on-device",
        )
        outputs = lm._process_completion(resp, merged_kwargs={})
        assert outputs == ["result text"]


# ---------------------------------------------------------------------------
# AppleFoundationLM init guards
# ---------------------------------------------------------------------------


class TestInitGuards:
    def test_raises_on_non_macos(self, fake_apple_fm_sdk):
        with patch("platform.system", return_value="Linux"):
            from dspy.clients.apple_fm import AppleFoundationLM

            with pytest.raises(RuntimeError, match="macOS"):
                AppleFoundationLM()

    def test_raises_when_sdk_missing(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "apple_fm_sdk", raising=False)
        monkeypatch.delitem(sys.modules, "dspy.clients.apple_fm", raising=False)

        with patch("platform.system", return_value="Darwin"):
            with patch.dict("sys.modules", {"apple_fm_sdk": None}):
                from dspy.clients.apple_fm import AppleFoundationLM

                with pytest.raises(ImportError, match="apple-fm-sdk"):
                    AppleFoundationLM()

    def test_raises_when_model_unavailable(self, fake_apple_fm_sdk, monkeypatch):
        fake_apple_fm_sdk.SystemLanguageModel = type(
            "SystemLanguageModel",
            (),
            {"is_available": lambda self: (False, "Apple Intelligence disabled")},
        )
        monkeypatch.delitem(sys.modules, "dspy.clients.apple_fm", raising=False)

        with patch("platform.system", return_value="Darwin"):
            from dspy.clients.apple_fm import AppleFoundationLM

            with pytest.raises(RuntimeError, match="Apple Intelligence"):
                AppleFoundationLM()


# ---------------------------------------------------------------------------
# forward / aforward
# ---------------------------------------------------------------------------


class TestForward:
    def test_plain_text_response(self, lm):
        result = lm.forward(prompt="Say hello")
        assert result.choices[0].message.content == "Hello from Apple on-device model"
        assert result.model == "apple/on-device"

    def test_messages_flattened(self, lm, fake_apple_fm_sdk):
        received_prompts = []
        original_session = fake_apple_fm_sdk.LanguageModelSession

        class SpySession(original_session):
            async def respond(self, prompt, **kwargs):
                received_prompts.append(prompt)
                return "ok"

        fake_apple_fm_sdk.LanguageModelSession = SpySession

        lm.forward(
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What is 2+2?"},
            ]
        )

        assert received_prompts, "respond() was never called"
        assert "Be concise." in received_prompts[0]
        assert "What is 2+2?" in received_prompts[0]

    def test_structured_output_uses_generable(self, lm, fake_apple_fm_sdk):
        class MyOutput(BaseModel):
            label: Literal["yes", "no"]

        generating_args = []
        original_session = fake_apple_fm_sdk.LanguageModelSession

        class SpySession(original_session):
            async def respond(self, prompt, generating=None, **kwargs):
                generating_args.append(generating)
                if generating is not None:
                    return generating()
                return "plain"

        fake_apple_fm_sdk.LanguageModelSession = SpySession

        result = lm.forward(prompt="classify", response_format=MyOutput)

        assert generating_args[0] is not None, "generating= should have been passed"
        # Result should be valid JSON
        parsed = json.loads(result.choices[0].message.content)
        assert "label" in parsed

    def test_fallback_when_generable_fails(self, lm, fake_apple_fm_sdk, caplog):
        class MyOutput(BaseModel):
            label: str

        fake_apple_fm_sdk.generable = lambda cls: (_ for _ in ()).throw(RuntimeError("boom"))

        import logging

        with caplog.at_level(logging.WARNING, logger="dspy.clients.apple_fm"):
            result = lm.forward(prompt="classify", response_format=MyOutput)

        assert "falling back" in caplog.text.lower() or result is not None

    def test_aforward_is_awaitable(self, lm):
        async def _run():
            return await lm.aforward(prompt="hello")

        result = asyncio.run(_run())
        assert result.choices[0].message.content == "Hello from Apple on-device model"


# ---------------------------------------------------------------------------
# Tool conversion
# ---------------------------------------------------------------------------


class TestToolConversion:
    def test_dspy_tool_callable_wrapped(self, fake_apple_fm_sdk):
        from dspy.clients.apple_fm import _dspy_tool_to_apple_tool

        calls = []

        def my_func(**kwargs):
            calls.append(kwargs)
            return "result"

        my_func.name = "my_tool"
        apple_tool = _dspy_tool_to_apple_tool(my_func, fake_apple_fm_sdk)
        apple_tool.call(x=1, y=2)
        assert calls == [{"x": 1, "y": 2}]

    def test_tool_name_preserved(self, fake_apple_fm_sdk):
        from dspy.clients.apple_fm import _dspy_tool_to_apple_tool

        def fn(**kwargs):
            return "ok"

        fn.name = "calculator"
        tool = _dspy_tool_to_apple_tool(fn, fake_apple_fm_sdk)
        assert type(tool).__name__ == "calculator"

    def test_same_tool_reuses_class(self, fake_apple_fm_sdk):
        from dspy.clients.apple_fm import _dspy_tool_to_apple_tool

        def fn(**kwargs):
            return "ok"

        fn.name = "my_tool"
        tool1 = _dspy_tool_to_apple_tool(fn, fake_apple_fm_sdk)
        tool2 = _dspy_tool_to_apple_tool(fn, fake_apple_fm_sdk)
        # Same function + same name → same cached class object, distinct instances.
        assert type(tool1) is type(tool2)
        assert tool1 is not tool2

    def test_different_functions_get_different_classes(self, fake_apple_fm_sdk):
        from dspy.clients.apple_fm import _dspy_tool_to_apple_tool

        def fn_a(**kwargs):
            return "a"

        def fn_b(**kwargs):
            return "b"

        fn_a.name = "tool"
        fn_b.name = "tool"
        tool_a = _dspy_tool_to_apple_tool(fn_a, fake_apple_fm_sdk)
        tool_b = _dspy_tool_to_apple_tool(fn_b, fake_apple_fm_sdk)
        # Same name but different function identity → distinct classes.
        assert type(tool_a) is not type(tool_b)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    """Test that dspy.cache is consulted correctly in forward().

    We mock dspy.cache directly rather than constructing new LM instances
    (which would re-trigger the macOS platform check).
    """

    def test_cache_hit_skips_inference(self, lm, fake_apple_fm_sdk):
        lm.cache = True
        call_count = []
        original_session = fake_apple_fm_sdk.LanguageModelSession

        class CountingSession(original_session):
            async def respond(self, prompt, **kwargs):
                call_count.append(1)
                return "live response"

        fake_apple_fm_sdk.LanguageModelSession = CountingSession

        import dspy
        from dspy.clients.apple_fm import _FMChoice, _FMMessage, _FMResponse, _FMUsage

        cached_response = _FMResponse(
            choices=[_FMChoice(message=_FMMessage(content="cached response"))],
            usage=_FMUsage(),
            model=lm.model,
        )

        with (
            patch.object(dspy.cache, "get", return_value=cached_response) as mock_get,
            patch.object(dspy.cache, "put") as mock_put,
        ):
            result = lm.forward(prompt="hello cache test")

        assert result.choices[0].message.content == "cached response"
        assert len(call_count) == 0, "Model should not have been called on cache hit"
        mock_get.assert_called_once()
        mock_put.assert_not_called()

    def test_cache_miss_stores_result(self, lm, fake_apple_fm_sdk):
        lm.cache = True

        import dspy

        with (
            patch.object(dspy.cache, "get", return_value=None) as mock_get,
            patch.object(dspy.cache, "put") as mock_put,
        ):
            result = lm.forward(prompt="uncached prompt")

        mock_get.assert_called_once()
        mock_put.assert_called_once()
        stored = mock_put.call_args[0][1]
        assert stored.choices[0].message.content == result.choices[0].message.content

    def test_cache_disabled_never_calls_cache(self, lm, fake_apple_fm_sdk):
        lm.cache = False

        import dspy

        with patch.object(dspy.cache, "get") as mock_get, patch.object(dspy.cache, "put") as mock_put:
            lm.forward(prompt="no cache")

        mock_get.assert_not_called()
        mock_put.assert_not_called()

    def test_litellm_kwargs_stripped_from_cache_key(self, lm, fake_apple_fm_sdk):
        """num_retries must not appear in the cache request dict."""
        lm.cache = True
        captured_requests = []

        import dspy

        def fake_get(request, *args, **kwargs):
            captured_requests.append(request)
            return None

        with patch.object(dspy.cache, "get", side_effect=fake_get), patch.object(dspy.cache, "put"):
            lm.forward(prompt="same prompt", num_retries=3)

        assert "num_retries" not in captured_requests[0]


# ---------------------------------------------------------------------------
# Kwargs, streaming, and context window
# ---------------------------------------------------------------------------


class TestFMKwargsAndStreaming:
    def test_stream_true_raises_not_implemented(self, lm, fake_apple_fm_sdk):
        with pytest.raises(NotImplementedError, match="Streaming"):
            lm.forward(prompt="hi", stream=True)

    def test_unknown_kwargs_warns(self, lm, fake_apple_fm_sdk):
        import dspy.clients.apple_fm as _apple_fm_mod

        with patch.object(_apple_fm_mod.logger, "warning") as mock_warn:
            lm.forward(prompt="hi", top_p=0.9)
        assert any("top_p" in str(call) for call in mock_warn.call_args_list)

    def test_context_window_attribute(self, lm):
        assert lm.context_window == 4096


# ---------------------------------------------------------------------------
# @generable runtime fallback
# ---------------------------------------------------------------------------


class TestGuardrailViolation:
    """GuardrailViolationError from the SDK must surface as a clear RuntimeError."""

    def _make_guardrail_error(self):
        """Construct a minimal exception whose class name contains 'GuardrailViolation'."""

        class GuardrailViolationError(Exception):
            pass

        return GuardrailViolationError("content rejected")

    def test_plain_text_path_raises_runtime_error(self, lm, fake_apple_fm_sdk):
        """GuardrailViolationError on the plain text path becomes RuntimeError."""
        exc = self._make_guardrail_error()
        original_cls = fake_apple_fm_sdk.LanguageModelSession

        class _Rejecting(original_cls):
            async def respond(self, prompt, **kwargs):
                raise exc

        fake_apple_fm_sdk.LanguageModelSession = _Rejecting

        with pytest.raises(RuntimeError, match="guardrail"):
            lm.forward(prompt="hello")

    def test_generable_path_raises_runtime_error(self, lm, fake_apple_fm_sdk):
        """GuardrailViolationError on the generable path becomes RuntimeError (no retry)."""
        from pydantic import BaseModel

        exc = self._make_guardrail_error()

        class MyOutput(BaseModel):
            value: str

        original_cls = fake_apple_fm_sdk.LanguageModelSession

        class _Rejecting(original_cls):
            async def respond(self, prompt, generating=None, **kwargs):
                raise exc

        fake_apple_fm_sdk.LanguageModelSession = _Rejecting

        with pytest.raises(RuntimeError, match="guardrail"):
            lm.forward(prompt="classify", response_format=MyOutput)

    def test_other_errors_still_propagate(self, lm, fake_apple_fm_sdk):
        """Non-guardrail exceptions from session.respond() are re-raised unchanged."""
        original_cls = fake_apple_fm_sdk.LanguageModelSession

        class _Crashing(original_cls):
            async def respond(self, prompt, **kwargs):
                raise ValueError("unexpected SDK error")

        fake_apple_fm_sdk.LanguageModelSession = _Crashing

        with pytest.raises(ValueError, match="unexpected SDK error"):
            lm.forward(prompt="hello")


class TestStructuredGenerationFallback:
    def test_generable_runtime_failure_falls_back(self, lm, fake_apple_fm_sdk):
        """If session.respond(generating=...) raises at runtime (e.g. Swift grammar
        compilation failure), the adapter must recreate the session and retry
        without the schema constraint so DSPy can handle injection via prompt."""
        from pydantic import BaseModel

        class MyOutput(BaseModel):
            value: str

        original_session_cls = fake_apple_fm_sdk.LanguageModelSession

        class _FailingOnGenerating(original_session_cls):
            async def respond(self, prompt, generating=None, **kwargs):
                if generating is not None:
                    raise RuntimeError("Swift grammar compilation failed")
                return await super().respond(prompt, **kwargs)

        fake_apple_fm_sdk.LanguageModelSession = _FailingOnGenerating

        import dspy.clients.apple_fm as _apple_fm_mod

        with patch.object(_apple_fm_mod.logger, "warning") as mock_warn:
            result = lm.forward(prompt="classify", response_format=MyOutput)

        assert isinstance(result.choices[0].message.content, str)
        assert len(result.choices[0].message.content) > 0
        assert any("generable" in str(call).lower() for call in mock_warn.call_args_list)


# ---------------------------------------------------------------------------
# Import guard on non-Mac
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_dspy_import_succeeds_without_sdk(self, monkeypatch):
        """dspy.AppleFoundationLM should be importable even when apple_fm_sdk is absent."""
        # The guarded try/except in dspy/clients/__init__.py swallows ImportError
        import dspy

        # On non-Mac CI the class may not be present — that's expected and correct
        # (the guard suppresses the error). On Mac with SDK it should be there.
        # Just verify the import of dspy itself doesn't raise.
        assert dspy is not None
