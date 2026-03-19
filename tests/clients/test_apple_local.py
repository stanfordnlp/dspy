"""Unit tests for AppleLocalLM (MLX backend).

All tests run on any platform by mocking ``mlx_lm`` and patching
``platform.system`` / ``platform.machine``.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fake mlx_lm module
# ---------------------------------------------------------------------------


def _make_fake_mlx_lm():
    mlx_lm = types.ModuleType("mlx_lm")

    # sample_utils submodule with make_sampler
    sample_utils = types.ModuleType("mlx_lm.sample_utils")

    def make_sampler(temp=0.0):
        return {"temp": temp}

    sample_utils.make_sampler = make_sampler
    mlx_lm.sample_utils = sample_utils

    class _FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|{role}|>{content}")
            if add_generation_prompt:
                parts.append("<|assistant|>")
            return "".join(parts)

        def encode(self, text):
            """Simple word-level tokenizer used for token-count tests."""
            return text.split()

    class _FakeModel:
        pass

    class _FakeGenerationResponse:
        def __init__(self, text):
            self.text = text

    def load(model_path):
        return _FakeModel(), _FakeTokenizer()

    def generate(model, tokenizer, prompt, max_tokens=100, sampler=None, verbose=False):
        return f"MLX response to: {prompt[:30]}"

    def stream_generate(model, tokenizer, prompt, max_tokens=100, sampler=None):
        for word in f"MLX response to: {prompt[:30]}".split():
            yield _FakeGenerationResponse(word + " ")

    mlx_lm.load = load
    mlx_lm.generate = generate
    mlx_lm.stream_generate = stream_generate
    return mlx_lm


@pytest.fixture(autouse=True)
def fake_mlx_lm(monkeypatch):
    mlx_lm = _make_fake_mlx_lm()
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", mlx_lm.sample_utils)
    monkeypatch.delitem(sys.modules, "dspy.clients.apple_local", raising=False)
    return mlx_lm


@pytest.fixture()
def lm(fake_mlx_lm):
    with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
        from dspy.clients.apple_local import AppleLocalLM

        return AppleLocalLM("mlx-community/Llama-3.2-3B-Instruct-4bit", cache=False)


# ---------------------------------------------------------------------------
# Init guards
# ---------------------------------------------------------------------------


class TestInitGuards:
    def test_raises_on_non_macos(self, fake_mlx_lm):
        with patch("platform.system", return_value="Linux"), patch("platform.machine", return_value="arm64"):
            from dspy.clients.apple_local import AppleLocalLM

            with pytest.raises(RuntimeError, match="macOS"):
                AppleLocalLM("mlx-community/some-model")

    def test_raises_on_intel_mac(self, fake_mlx_lm):
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="x86_64"):
            from dspy.clients.apple_local import AppleLocalLM

            with pytest.raises(RuntimeError, match="arm64"):
                AppleLocalLM("mlx-community/some-model")

    def test_raises_on_unknown_backend(self, fake_mlx_lm):
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
            from dspy.clients.apple_local import AppleLocalLM

            with pytest.raises(ValueError, match="Unknown backend"):
                AppleLocalLM("mlx-community/some-model", backend="tpu")

    def test_coreml_raises_not_implemented(self, fake_mlx_lm):
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
            from dspy.clients.apple_local import AppleLocalLM

            with pytest.raises(NotImplementedError, match="CoreML"):
                AppleLocalLM("model.mlpackage", backend="coreml")

    def test_raises_when_mlx_missing(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "mlx_lm", raising=False)
        monkeypatch.delitem(sys.modules, "dspy.clients.apple_local", raising=False)
        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch.dict("sys.modules", {"mlx_lm": None}),
        ):
            from dspy.clients.apple_local import AppleLocalLM

            with pytest.raises(ImportError, match="mlx-lm"):
                AppleLocalLM("mlx-community/some-model")


# ---------------------------------------------------------------------------
# Chat template
# ---------------------------------------------------------------------------


class TestChatTemplate:
    def test_apply_chat_template_used_when_available(self, lm, fake_mlx_lm):
        calls = []
        original = lm._mlx_tokenizer.apply_chat_template

        def spy(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        lm._mlx_tokenizer.apply_chat_template = spy
        lm.forward(messages=[{"role": "user", "content": "hello"}])
        assert calls, "apply_chat_template should have been called"

    def test_bare_prompt_wrapped_as_user_message(self, lm, fake_mlx_lm):
        received = []
        original_generate = fake_mlx_lm.generate

        def spy_generate(model, tokenizer, prompt, **kwargs):
            received.append(prompt)
            return original_generate(model, tokenizer, prompt, **kwargs)

        fake_mlx_lm.generate = spy_generate

        lm.forward(prompt="What is 2+2?")
        assert received, "generate() should have been called"
        # The prompt should contain the user content
        assert "What is 2+2?" in received[0]

    def test_fallback_when_no_chat_template(self, lm, fake_mlx_lm):
        # Replace the tokenizer with a bare object that has no apply_chat_template.
        # encode() must still exist because forward() calls it for token counting.
        class _BareTokenizer:
            def encode(self, text):
                return text.split()

        lm._mlx_tokenizer = _BareTokenizer()

        result = lm.forward(
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hi"},
            ]
        )
        assert result.choices[0].message.content  # still got a response


# ---------------------------------------------------------------------------
# forward / aforward
# ---------------------------------------------------------------------------


class TestForward:
    def test_returns_fm_response(self, lm):
        from dspy.clients.apple_fm import _FMResponse

        result = lm.forward(prompt="hello")
        assert isinstance(result, _FMResponse)
        assert result.model == "mlx-community/Llama-3.2-3B-Instruct-4bit"

    def test_response_text_is_string(self, lm):
        result = lm.forward(prompt="hello")
        assert isinstance(result.choices[0].message.content, str)
        assert len(result.choices[0].message.content) > 0

    def test_process_completion_compatible(self, lm):
        """_FMResponse from AppleLocalLM must work with BaseLM._process_completion."""
        result = lm.forward(prompt="test")
        outputs = lm._process_completion(result, merged_kwargs={})
        assert len(outputs) == 1
        assert isinstance(outputs[0], str)

    def test_temperature_kwarg_forwarded(self, lm, fake_mlx_lm):
        received_kwargs = {}
        original_generate = fake_mlx_lm.generate

        def spy(model, tokenizer, prompt, **kwargs):
            received_kwargs.update(kwargs)
            return original_generate(model, tokenizer, prompt, **kwargs)

        fake_mlx_lm.generate = spy
        lm.forward(prompt="test", temperature=0.9)
        assert received_kwargs.get("sampler") is not None

    def test_max_tokens_kwarg_forwarded(self, lm, fake_mlx_lm):
        received_kwargs = {}
        original_generate = fake_mlx_lm.generate

        def spy(model, tokenizer, prompt, **kwargs):
            received_kwargs.update(kwargs)
            return original_generate(model, tokenizer, prompt, **kwargs)

        fake_mlx_lm.generate = spy
        lm.forward(prompt="test", max_tokens=256)
        assert received_kwargs.get("max_tokens") == 256

    def test_aforward_is_awaitable(self, lm):
        async def _run():
            return await lm.aforward(prompt="hello async")

        result = asyncio.run(_run())
        assert isinstance(result.choices[0].message.content, str)

    def test_usage_dict_conversion(self, lm):
        result = lm.forward(prompt="hello world")
        usage = dict(result.usage)
        assert set(usage.keys()) == {"prompt_tokens", "completion_tokens", "total_tokens"}
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


# ---------------------------------------------------------------------------
# Concurrency, tools, and usage
# ---------------------------------------------------------------------------


class TestLocalConcurrencyAndTools:
    def test_tools_raises_not_implemented(self, lm):
        with pytest.raises(NotImplementedError, match="AppleFoundationLM"):
            lm.forward(prompt="hi", tools=[object()])

    def test_usage_counts_nonzero(self, lm):
        result = lm.forward(prompt="count my tokens please")
        assert result.usage.prompt_tokens > 0
        assert result.usage.completion_tokens > 0
        assert result.usage.total_tokens == (result.usage.prompt_tokens + result.usage.completion_tokens)

    def test_hidden_params_response_cost_zero(self, lm):
        result = lm.forward(prompt="cost check")
        assert result._hidden_params.get("response_cost") == 0.0

    def test_concurrent_aforward_serialized(self, lm):
        import threading
        import time

        concurrent_count = 0
        max_concurrent = 0
        _lock = threading.Lock()
        original_forward = lm.forward

        def tracking_forward(prompt=None, messages=None, **kwargs):
            nonlocal concurrent_count, max_concurrent
            with _lock:
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
            time.sleep(0.02)  # Hold long enough for overlap to show if unserialized
            result = original_forward(prompt=prompt, messages=messages, **kwargs)
            with _lock:
                concurrent_count -= 1
            return result

        async def _run():
            lm._semaphore = None  # Reset so a fresh semaphore is created
            lm.forward = tracking_forward
            try:
                await asyncio.gather(
                    lm.aforward(prompt="a"),
                    lm.aforward(prompt="b"),
                    lm.aforward(prompt="c"),
                )
            finally:
                lm.forward = original_forward

        asyncio.run(_run())
        assert max_concurrent == 1, f"Expected max 1 concurrent MLX call, got {max_concurrent}"

    def test_stream_true_raises_not_implemented(self, lm):
        with pytest.raises(NotImplementedError, match="streamify"):
            lm.forward(prompt="hi", stream=True)

    def test_unknown_kwargs_warns(self, lm):
        import dspy.clients.apple_local as _apple_local_mod

        with patch.object(_apple_local_mod.logger, "warning") as mock_warn:
            lm.forward(prompt="hi", top_p=0.9)
        assert any("top_p" in str(call) for call in mock_warn.call_args_list)

    def test_context_window_attribute(self, lm):
        assert isinstance(lm.context_window, int)
        assert lm.context_window > 0

    def test_context_window_warning_on_long_prompt(self, lm, fake_mlx_lm):
        # Make encode() return a very long token list to simulate context overflow.
        original_encode = lm._mlx_tokenizer.encode

        def long_encode(text):
            # Return enough tokens to exceed context_window - max_tokens
            return list(range(lm.context_window))

        lm._mlx_tokenizer.encode = long_encode
        import dspy.clients.apple_local as _apple_local_mod

        try:
            with patch.object(_apple_local_mod.logger, "warning") as mock_warn:
                lm.forward(prompt="this prompt is very long")
        finally:
            lm._mlx_tokenizer.encode = original_encode

        assert any("context_window" in str(call) or "truncated" in str(call) for call in mock_warn.call_args_list)


# ---------------------------------------------------------------------------
# bits parameter
# ---------------------------------------------------------------------------


class TestBitsParameter:
    def test_bits_stored_on_instance(self, fake_mlx_lm):
        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
            from dspy.clients.apple_local import AppleLocalLM

            lm = AppleLocalLM("mlx-community/some-4bit-model", bits=4, cache=False)
        assert lm._bits == 4

    def test_bits_none_by_default(self, lm):
        assert lm._bits is None


# ---------------------------------------------------------------------------
# max_concurrency warning
# ---------------------------------------------------------------------------


class TestMaxConcurrencyWarning:
    def test_max_concurrency_gt_1_warns(self, fake_mlx_lm):
        import dspy.clients.apple_local as _apple_local_mod

        with (
            patch("platform.system", return_value="Darwin"),
            patch("platform.machine", return_value="arm64"),
            patch.object(_apple_local_mod.logger, "warning") as mock_warn,
        ):
            from dspy.clients.apple_local import AppleLocalLM

            AppleLocalLM(
                "mlx-community/Llama-3.2-3B-Instruct-4bit",
                max_concurrency=2,
                cache=False,
            )

        assert any("max_concurrency" in str(call) for call in mock_warn.call_args_list)

    def test_max_concurrency_1_no_warning(self, fake_mlx_lm, caplog):
        import logging

        with patch("platform.system", return_value="Darwin"), patch("platform.machine", return_value="arm64"):
            from dspy.clients.apple_local import AppleLocalLM

            with caplog.at_level(logging.WARNING, logger="dspy.clients.apple_local"):
                AppleLocalLM(
                    "mlx-community/Llama-3.2-3B-Instruct-4bit",
                    max_concurrency=1,
                    cache=False,
                )

        assert not any("max_concurrency" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_aforward_sends_chunks_to_send_stream(self, lm):
        """aforward() sends _LocalStreamChunk objects when send_stream is active."""
        from unittest.mock import AsyncMock

        import dspy
        from dspy.clients.apple_local import _LocalStreamChunk

        sent_chunks = []
        mock_send_stream = AsyncMock()
        mock_send_stream.send = AsyncMock(side_effect=lambda c: sent_chunks.append(c))

        async def _run():
            with dspy.context(send_stream=mock_send_stream):
                return await lm.aforward(prompt="hello streaming")

        response = asyncio.run(_run())
        assert any(isinstance(c, _LocalStreamChunk) for c in sent_chunks), (
            "Expected _LocalStreamChunk objects sent to send_stream"
        )
        assert response.choices[0].message.content, "Expected non-empty response"

    def test_stream_chunks_carry_text(self, lm):
        """Each chunk has non-empty .text and .model fields."""
        from unittest.mock import AsyncMock

        import dspy
        from dspy.clients.apple_local import _LocalStreamChunk

        sent_chunks = []
        mock_send_stream = AsyncMock()
        mock_send_stream.send = AsyncMock(side_effect=lambda c: sent_chunks.append(c))

        async def _run():
            with dspy.context(send_stream=mock_send_stream):
                await lm.aforward(prompt="hi")

        asyncio.run(_run())
        stream_chunks = [c for c in sent_chunks if isinstance(c, _LocalStreamChunk)]
        assert all(isinstance(c.text, str) and len(c.text) > 0 for c in stream_chunks)
        assert all(c.model == lm.model for c in stream_chunks)

    def test_aforward_without_send_stream_uses_blocking_path(self, lm):
        """aforward() without send_stream still returns a valid response."""

        async def _run():
            return await lm.aforward(prompt="hello")

        result = asyncio.run(_run())
        assert result.choices[0].message.content

    def test_stream_full_text_equals_concatenated_chunks(self, lm):
        """The response text equals the concatenation of all chunk texts."""
        from unittest.mock import AsyncMock

        import dspy
        from dspy.clients.apple_local import _LocalStreamChunk

        sent_chunks = []
        mock_send_stream = AsyncMock()
        mock_send_stream.send = AsyncMock(side_effect=lambda c: sent_chunks.append(c))

        async def _run():
            with dspy.context(send_stream=mock_send_stream):
                return await lm.aforward(prompt="concatenation test")

        response = asyncio.run(_run())
        concatenated = "".join(c.text for c in sent_chunks if isinstance(c, _LocalStreamChunk))
        assert response.choices[0].message.content == concatenated

    def test_stream_true_in_forward_still_raises(self, lm):
        with pytest.raises(NotImplementedError, match="streamify"):
            lm.forward(prompt="hi", stream=True)


class TestImportGuard:
    def test_dspy_importable_without_mlx(self, monkeypatch):
        import dspy

        assert dspy is not None
