"""Tests for FabricAzureOpenAI client."""

from unittest.mock import MagicMock, patch

import pytest

from dspy.clients.fabric_azure_openai import FabricAzureOpenAI


class TestFabricAzureOpenAI:
    """Test suite for FabricAzureOpenAI class."""

    def test_initialization(self):
        """Test that FabricAzureOpenAI can be initialized."""
        lm = FabricAzureOpenAI(deployment_name="gpt-4o")
        assert lm.deployment_name == "gpt-4o"
        assert lm.model == "gpt-4o"
        assert lm.is_reasoning_model is False

    def test_reasoning_model_detection_o1(self):
        """Test that o1 models are detected as reasoning models."""
        lm = FabricAzureOpenAI(deployment_name="o1-preview")
        assert lm.is_reasoning_model is True

    def test_reasoning_model_detection_o3(self):
        """Test that o3 models are detected as reasoning models."""
        lm = FabricAzureOpenAI(deployment_name="o3-mini")
        assert lm.is_reasoning_model is True

    def test_reasoning_model_detection_gpt5(self):
        """Test that gpt-5 models are detected as reasoning models."""
        lm = FabricAzureOpenAI(deployment_name="gpt-5")
        assert lm.is_reasoning_model is True

    def test_standard_model_detection(self):
        """Test that standard models are not detected as reasoning models."""
        lm = FabricAzureOpenAI(deployment_name="gpt-4o")
        assert lm.is_reasoning_model is False

    def test_prepare_messages_with_string_prompt(self):
        """Test message preparation with a string prompt."""
        lm = FabricAzureOpenAI(deployment_name="gpt-4o")
        messages = lm._prepare_messages("Hello, world!", None)
        assert messages == [{"role": "user", "content": "Hello, world!"}]

    def test_prepare_messages_with_list_prompt(self):
        """Test message preparation with a list of messages."""
        lm = FabricAzureOpenAI(deployment_name="gpt-4o")
        prompt = [{"role": "user", "content": "Hello"}]
        messages = lm._prepare_messages(prompt, None)
        assert messages == prompt

    def test_prepare_messages_with_messages_param(self):
        """Test message preparation with messages parameter."""
        lm = FabricAzureOpenAI(deployment_name="gpt-4o")
        messages_input = [{"role": "user", "content": "Test"}]
        messages = lm._prepare_messages(None, messages_input)
        assert messages == messages_input

    def test_build_payload_standard_model(self):
        """Test payload building for standard models."""
        lm = FabricAzureOpenAI(deployment_name="gpt-4o", max_tokens=1000, temperature=0.5)
        messages = [{"role": "user", "content": "Hello"}]
        payload = lm._build_payload(messages)

        assert payload["messages"] == messages
        assert payload["max_tokens"] == 1000
        assert payload["temperature"] == 0.5
        assert payload["n"] == 1
        assert "max_completion_tokens" not in payload

    def test_build_payload_reasoning_model(self):
        """Test payload building for reasoning models."""
        lm = FabricAzureOpenAI(deployment_name="o1-preview", max_tokens=2000)
        messages = [{"role": "user", "content": "Solve this"}]
        payload = lm._build_payload(messages)

        assert payload["messages"] == messages
        assert payload["max_completion_tokens"] == 2000
        assert "temperature" not in payload
        assert "n" not in payload
        assert "max_tokens" not in payload

    def test_build_payload_with_override_kwargs(self):
        """Test that kwargs can override default values."""
        lm = FabricAzureOpenAI(deployment_name="gpt-4o", max_tokens=1000, temperature=0.0)
        messages = [{"role": "user", "content": "Hello"}]
        payload = lm._build_payload(messages, max_tokens=2000, temperature=0.8)

        assert payload["max_tokens"] == 2000
        assert payload["temperature"] == 0.8

    def test_get_fabric_config_import_error(self):
        """Test that appropriate error is raised when Fabric SDK is not available."""
        lm = FabricAzureOpenAI(deployment_name="gpt-4o")

        with pytest.raises(ImportError) as exc_info:
            lm._get_fabric_config()

        assert "Microsoft Fabric SDK packages are required" in str(exc_info.value)

    @patch("dspy.clients.fabric_azure_openai.requests.post")
    def test_make_request_success(self, mock_post):
        """Test successful API request."""
        # Mock Fabric SDK imports
        mock_fabric_config = MagicMock()
        mock_fabric_config.ml_workload_endpoint = "https://test.endpoint/"

        with patch(
            "dspy.clients.fabric_azure_openai.FabricAzureOpenAI._get_fabric_config",
            return_value=(mock_fabric_config, "Bearer test-token"),
        ):
            # Mock successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [
                    {"message": {"content": "Hello there!"}},
                    {"message": {"content": "How are you?"}},
                ]
            }
            mock_post.return_value = mock_response

            lm = FabricAzureOpenAI(deployment_name="gpt-4o")
            payload = {"messages": [{"role": "user", "content": "Hi"}]}
            result = lm._make_request(payload)

            assert result == ["Hello there!", "How are you?"]
            assert mock_post.called

    @patch("dspy.clients.fabric_azure_openai.requests.post")
    def test_make_request_failure(self, mock_post):
        """Test failed API request."""
        # Mock Fabric SDK imports
        mock_fabric_config = MagicMock()
        mock_fabric_config.ml_workload_endpoint = "https://test.endpoint/"

        with patch(
            "dspy.clients.fabric_azure_openai.FabricAzureOpenAI._get_fabric_config",
            return_value=(mock_fabric_config, "Bearer test-token"),
        ):
            # Mock failed response
            mock_response = MagicMock()
            mock_response.status_code = 400
            mock_response.text = "Bad request"
            mock_post.return_value = mock_response

            lm = FabricAzureOpenAI(deployment_name="gpt-4o")
            payload = {"messages": [{"role": "user", "content": "Hi"}]}

            with pytest.raises(Exception) as exc_info:
                lm._make_request(payload)

            assert "API call failed: 400" in str(exc_info.value)

    @patch("dspy.clients.fabric_azure_openai.requests.post")
    def test_forward_success(self, mock_post):
        """Test forward method with successful response."""
        # Mock Fabric SDK imports
        mock_fabric_config = MagicMock()
        mock_fabric_config.ml_workload_endpoint = "https://test.endpoint/"

        with patch(
            "dspy.clients.fabric_azure_openai.FabricAzureOpenAI._get_fabric_config",
            return_value=(mock_fabric_config, "Bearer test-token"),
        ):
            # Mock successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"choices": [{"message": {"content": "Test response"}}]}
            mock_post.return_value = mock_response

            lm = FabricAzureOpenAI(deployment_name="gpt-4o")
            result = lm.forward(prompt="Test prompt")

            # Check that result has the expected structure
            assert hasattr(result, "choices")
            assert len(result.choices) == 1
            assert result.choices[0].message.content == "Test response"
            assert result.choices[0].message.role == "assistant"
            assert result.model == "gpt-4o"

    @patch("dspy.clients.fabric_azure_openai.requests.post")
    def test_call_with_prompt(self, mock_post):
        """Test __call__ method with prompt."""
        # Mock Fabric SDK imports
        mock_fabric_config = MagicMock()
        mock_fabric_config.ml_workload_endpoint = "https://test.endpoint/"

        with patch(
            "dspy.clients.fabric_azure_openai.FabricAzureOpenAI._get_fabric_config",
            return_value=(mock_fabric_config, "Bearer test-token"),
        ):
            # Mock successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"choices": [{"message": {"content": "Response text"}}]}
            mock_post.return_value = mock_response

            lm = FabricAzureOpenAI(deployment_name="gpt-4o")
            result = lm(prompt="Hello")

            assert result == ["Response text"]

    @patch("dspy.clients.fabric_azure_openai.requests.post")
    def test_call_with_messages(self, mock_post):
        """Test __call__ method with messages."""
        # Mock Fabric SDK imports
        mock_fabric_config = MagicMock()
        mock_fabric_config.ml_workload_endpoint = "https://test.endpoint/"

        with patch(
            "dspy.clients.fabric_azure_openai.FabricAzureOpenAI._get_fabric_config",
            return_value=(mock_fabric_config, "Bearer test-token"),
        ):
            # Mock successful response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"choices": [{"message": {"content": "Response text"}}]}
            mock_post.return_value = mock_response

            lm = FabricAzureOpenAI(deployment_name="gpt-4o")
            messages = [{"role": "user", "content": "Hello"}]
            result = lm(messages=messages)

            assert result == ["Response text"]

    def test_call_without_prompt_or_messages(self):
        """Test that __call__ raises error when neither prompt nor messages provided."""
        lm = FabricAzureOpenAI(deployment_name="gpt-4o")

        with pytest.raises(ValueError) as exc_info:
            lm()

        assert "Either 'prompt' or 'messages' must be provided" in str(exc_info.value)

    def test_forward_without_prompt_or_messages(self):
        """Test that forward raises error when neither prompt nor messages provided."""
        lm = FabricAzureOpenAI(deployment_name="gpt-4o")

        with pytest.raises(ValueError) as exc_info:
            lm.forward()

        assert "Either 'prompt' or 'messages' must be provided" in str(exc_info.value)
