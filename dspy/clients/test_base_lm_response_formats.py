"""
Unit tests for _process_response method handling both dict and object formats.
Tests the fix for issue #8958 - web_search tools return dict format.
"""

import pytest

from dspy.clients.base_lm import BaseLM


class MockContent:
    """Mock content object (object format)"""
    def __init__(self, text):
        self.text = text


class MockOutputItem:
    """Mock output item (object format - without web_search)"""
    def __init__(self, item_type, content=None, summary=None):
        self.type = item_type
        if content:
            self.content = content
        if summary:
            self.summary = summary

    def model_dump(self):
        return {"type": self.type, "name": "test_function", "arguments": "{}"}


class MockResponse:
    """Mock response object"""
    def __init__(self, output):
        self.output = output
        self.usage = type("obj", (object,), {
            "completion_tokens": 10,
            "prompt_tokens": 5,
            "total_tokens": 15
        })()
        self.model = "gpt-4"


class TestProcessResponseFormats:
    """Test _process_response handles both dict and object formats"""

    @pytest.fixture
    def base_lm(self):
        """Create a BaseLM instance for testing"""
        return BaseLM(model="test-model", model_type="responses")

    def test_object_format_message(self, base_lm):
        """Test processing object format (normal responses without web_search)"""
        # Create mock response with object format
        mock_response = MockResponse(
            output=[
                MockOutputItem("message", content=[MockContent("Hello world")])
            ]
        )

        result = base_lm._process_response(mock_response)

        assert len(result) == 1
        assert result[0]["text"] == "Hello world"

    def test_dict_format_message(self, base_lm):
        """Test processing dict format (responses with web_search tools)"""
        # Create mock response with dict format (as returned by web_search)
        mock_response = MockResponse(
            output=[
                {
                    "type": "message",
                    "content": [{"text": "Hello from web search"}]
                }
            ]
        )

        result = base_lm._process_response(mock_response)

        assert len(result) == 1
        assert result[0]["text"] == "Hello from web search"

    def test_dict_format_with_multiple_content(self, base_lm):
        """Test dict format with multiple content items"""
        mock_response = MockResponse(
            output=[
                {
                    "type": "message",
                    "content": [
                        {"text": "Part 1"},
                        {"text": " Part 2"},
                        {"text": " Part 3"}
                    ]
                }
            ]
        )

        result = base_lm._process_response(mock_response)

        assert len(result) == 1
        assert result[0]["text"] == "Part 1 Part 2 Part 3"

    def test_object_format_function_call(self, base_lm):
        """Test function call in object format"""
        mock_item = MockOutputItem("function_call")
        mock_response = MockResponse(output=[mock_item])

        result = base_lm._process_response(mock_response)

        assert len(result) == 1
        assert "tool_calls" in result[0]
        assert len(result[0]["tool_calls"]) == 1

    def test_dict_format_function_call(self, base_lm):
        """Test function call in dict format"""
        mock_response = MockResponse(
            output=[
                {
                    "type": "function_call",
                    "name": "web_search",
                    "arguments": '{"query": "test"}'
                }
            ]
        )

        result = base_lm._process_response(mock_response)

        assert len(result) == 1
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["name"] == "web_search"

    def test_object_format_reasoning(self, base_lm):
        """Test reasoning content in object format"""
        mock_response = MockResponse(
            output=[
                MockOutputItem("reasoning", content=[MockContent("Thinking step 1")])
            ]
        )

        result = base_lm._process_response(mock_response)

        assert len(result) == 1
        assert "reasoning_content" in result[0]
        assert result[0]["reasoning_content"] == "Thinking step 1"

    def test_dict_format_reasoning(self, base_lm):
        """Test reasoning content in dict format"""
        mock_response = MockResponse(
            output=[
                {
                    "type": "reasoning",
                    "content": [{"text": "Reasoning step 1"}]
                }
            ]
        )

        result = base_lm._process_response(mock_response)

        assert len(result) == 1
        assert "reasoning_content" in result[0]
        assert result[0]["reasoning_content"] == "Reasoning step 1"

    def test_dict_format_reasoning_with_summary(self, base_lm):
        """Test reasoning with summary (fallback when no content)"""
        mock_response = MockResponse(
            output=[
                {
                    "type": "reasoning",
                    "summary": [{"text": "Summary text"}]
                }
            ]
        )

        result = base_lm._process_response(mock_response)

        assert len(result) == 1
        assert "reasoning_content" in result[0]
        assert result[0]["reasoning_content"] == "Summary text"

    def test_mixed_format_backwards_compatibility(self, base_lm):
        """Test that both formats can coexist (edge case)"""
        # Mix of object and dict formats in same response
        mock_response = MockResponse(
            output=[
                MockOutputItem("message", content=[MockContent("Object format")]),
                {"type": "message", "content": [{"text": " Dict format"}]}
            ]
        )

        result = base_lm._process_response(mock_response)

        assert len(result) == 1
        assert result[0]["text"] == "Object format Dict format"

    def test_empty_content(self, base_lm):
        """Test handling of empty content"""
        mock_response = MockResponse(
            output=[
                {"type": "message", "content": []}
            ]
        )

        result = base_lm._process_response(mock_response)

        assert len(result) == 1
        assert "text" not in result[0]  # No text key when no content
