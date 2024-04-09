import pytest
from unittest.mock import patch, MagicMock
import json
from dsp.modules.gpt4vision import (
    cached_gpt4vision_chat_request_wrapped,
    cached_gpt4vision_completion_request_wrapped,
    CacheMemory,
    NotebookCacheMemory,
    GPT4Vision,
    ERRORS
)
from dsp.primitives.vision import Image
import numpy as np



mock_chat_response = {
    "id": "chat-test",
    "object": "chat_completion",
    "created": 123456789,
    "model": "gpt-4-vision-preview",
    "choices": [{
        "message": {"content": "Test chat"},
        "finish_reason": "stop",
        "logprobs": None  # or the appropriate value if needed
    }],
}

mock_completion_response = {
    "id": "completion-test",
    "object": "completion",
    "created": 123456789,
    "model": "gpt-4-vision-preview",
    "choices": [{
        "message": {"content": "Test completion"},
        "finish_reason": "stop",
        "logprobs": None  # or the appropriate value if needed
    }],
}

# Test the basic_request method
def test_basic_chat_request():
  with patch('dsp.modules.gpt4vision.chat_request') as mock_chat_request:
      mock_chat_request.return_value = mock_chat_response
      gpt4vision = GPT4Vision()
      response = gpt4vision.basic_request(prompt="Test prompt")
      assert response == mock_chat_response
      mock_chat_request.assert_called_once()
      
# Test the request method with backoff and error handling
def test_request_with_backoff():
    with patch('dsp.modules.gpt4vision.GPT4Vision.basic_request') as mock_basic_request:
        mock_basic_request.side_effect = [ERRORS[0]("Rate limit exceeded", response=MagicMock(), body='{"error": {"message": "Rate limit exceeded", "code": 429}}'), mock_chat_response]
        gpt4vision = GPT4Vision()
        response = gpt4vision.request(prompt="Test prompt")
        assert response == mock_chat_response
        assert mock_basic_request.call_count == 2

# Test the __call__ method
def test_call_method():
  client = GPT4Vision()
  client.request("Test prompt")
  with patch('dsp.modules.gpt4vision.GPT4Vision.request') as mock_request:
    mock_request.return_value = mock_chat_response
    gpt4vision = GPT4Vision()
    completions = gpt4vision(prompt="Test prompt")
    assert completions == ["Test chat"]
    mock_request.assert_called_once()

# Test the logging functionality
def test_log_usage():
    with patch('logging.info') as mock_logging_info:
        gpt4vision = GPT4Vision()
        mock_response = {'usage': {'total_tokens': 100}}
        gpt4vision.log_usage(mock_response)
        mock_logging_info.assert_called_with("100")

# Test the completion request caching
# Test the chat request caching
def test_cached_gpt4vision_chat_request_wrapped():
    CacheMemory.clear()
    with patch('openai.chat.completions.create') as mock_chat_create:
        mock_chat_create.return_value = mock_chat_response
        messages = [{"role": "user", "content": "Message new"}]
        stringified = json.dumps({"messages": messages})

        # First call should use the API
        response1 = cached_gpt4vision_chat_request_wrapped(stringified_request=stringified)
        assert response1 == mock_chat_response
        mock_chat_create.assert_called_once()

        # Second call should use the cache and not call the API again
        response2 = cached_gpt4vision_chat_request_wrapped(stringified_request=stringified)
        assert response2 == mock_chat_response
        mock_chat_create.assert_called_once()

def test_cached_gpt4vision_completions_request_wrapped():
    CacheMemory.clear()
    with patch('openai.completions.create') as mock_chat_create:
        mock_chat_create.return_value = mock_completion_response
        messages = [{"role": "user", "content": "Message new"}]
        stringified = json.dumps({"prompt": messages})

        # First call should use the API
        response1 = cached_gpt4vision_completion_request_wrapped(stringified_request=stringified)
        assert response1 == mock_completion_response
        mock_chat_create.assert_called_once()

        # Second call should use the cache and not call the API again
        response2 = cached_gpt4vision_completion_request_wrapped(stringified_request=stringified)
        assert response2 == mock_completion_response
        mock_chat_create.assert_called_once()

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
