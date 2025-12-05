import pytest

import dspy


class TestHistoryModeDetection:
    """Tests for History mode auto-detection from message structure."""

    def test_flat_is_default_mode(self):
        """Messages with arbitrary keys default to flat mode."""
        history = dspy.History(messages=[{"question": "...", "answer": "..."}])
        assert history.mode == "flat"

    def test_detects_demo_mode(self):
        """Messages with only input_fields/output_fields are detected as demo mode."""
        history = dspy.History(messages=[{"input_fields": {"a": 1}, "output_fields": {"b": 2}}])
        assert history.mode == "demo"

    def test_detects_demo_mode_input_only(self):
        """Messages with only input_fields are detected as demo mode."""
        history = dspy.History(messages=[{"input_fields": {"a": 1}}])
        assert history.mode == "demo"

    def test_detects_raw_mode(self):
        """Messages with role+content are detected as raw mode."""
        history = dspy.History(messages=[{"role": "user", "content": "hello"}])
        assert history.mode == "raw"

    def test_detects_raw_mode_with_tool_calls(self):
        """Raw mode detected for tool_calls messages."""
        history = dspy.History(messages=[
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "test", "arguments": "{}"}}]}
        ])
        assert history.mode == "raw"

    def test_flat_with_extra_keys_beyond_role_content(self):
        """Messages with role+content AND extra keys fallback to flat mode."""
        history = dspy.History(messages=[{"role": "user", "content": "hello", "extra": "data"}])
        assert history.mode == "flat"

    def test_flat_with_input_fields_and_extra_keys(self):
        """Messages with input_fields AND extra keys fallback to flat mode."""
        history = dspy.History(messages=[{"question": "...", "input_fields": {"a": 1}}])
        assert history.mode == "flat"


class TestHistoryExplicitMode:
    """Tests for explicitly setting History mode."""

    def test_explicit_mode_overrides_auto_detection(self):
        """Explicit mode overrides auto-detection."""
        history = dspy.History(messages=[{"question": "...", "answer": "..."}], mode="signature")
        assert history.mode == "signature"

    def test_explicit_demo_mode(self):
        """Explicit mode='demo' sets demo mode."""
        history = dspy.History(messages=[{"input_fields": {"a": 1}}], mode="demo")
        assert history.mode == "demo"

    def test_explicit_raw_mode(self):
        """Explicit mode='raw' sets raw mode."""
        history = dspy.History(messages=[{"role": "user", "content": "hello"}], mode="raw")
        assert history.mode == "raw"

    def test_explicit_signature_mode(self):
        """Explicit mode='signature' sets signature mode."""
        history = dspy.History(messages=[{"question": "...", "answer": "..."}], mode="signature")
        assert history.mode == "signature"


class TestHistoryValidation:
    """Tests for History message validation."""

    def test_demo_mode_requires_dict_input_fields(self):
        """Demo mode with non-dict input_fields raises ValueError."""
        with pytest.raises(ValueError, match="'input_fields' must be a dict"):
            dspy.History(messages=[{"input_fields": "not a dict"}])

    def test_raw_mode_requires_string_or_list_or_none_content(self):
        """Raw mode with invalid content type raises ValueError."""
        with pytest.raises(ValueError, match="'content' must be a string, list, or None"):
            dspy.History(messages=[{"role": "user", "content": 123}])

    def test_raw_mode_allows_none_content(self):
        """Raw mode allows None content for tool call messages."""
        history = dspy.History(messages=[
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "test", "arguments": "{}"}}]}
        ])
        assert history.messages[0]["content"] is None

    def test_raw_mode_requires_string_role(self):
        """Raw mode with non-string role raises ValueError."""
        with pytest.raises(ValueError, match="'role' must be a string"):
            dspy.History(messages=[{"role": 123, "content": "hello"}])

    def test_raw_mode_allows_multimodal_content(self):
        """Raw mode allows list content for multimodal messages."""
        history = dspy.History(messages=[
            {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "image_url", "image_url": {"url": "..."}}]},
        ])
        assert history.mode == "raw"
        assert isinstance(history.messages[0]["content"], list)


class TestHistoryWithMessages:
    """Tests for History.with_messages() method."""

    def test_with_messages_preserves_mode(self):
        """with_messages() preserves mode and validates new messages."""
        base = dspy.History(messages=[{"role": "user", "content": "hi"}])
        extended = base.with_messages([{"role": "assistant", "content": "hello"}])
        assert extended.mode == "raw"
        assert len(extended.messages) == 2
        assert extended.messages[1]["content"] == "hello"

    def test_with_messages_validates_new_messages(self):
        """with_messages() validates appended messages against the mode."""
        base = dspy.History(messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(ValueError, match="'content' must be a string"):
            base.with_messages([{"role": "assistant", "content": 123}])


class TestHistoryFactoryMethods:
    """Tests for History factory methods."""

    def test_from_raw_creates_raw_mode(self):
        """from_raw() creates History with raw mode."""
        history = dspy.History.from_raw([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ])
        assert history.mode == "raw"
        assert len(history.messages) == 2

    def test_from_demos_creates_demo_mode(self):
        """from_demos() creates History with demo mode."""
        history = dspy.History.from_demos([
            {"input_fields": {"question": "2+2?"}, "output_fields": {"answer": "4"}},
        ])
        assert history.mode == "demo"
        assert len(history.messages) == 1

    def test_from_signature_pairs_creates_signature_mode(self):
        """from_signature_pairs() creates History with signature mode."""
        history = dspy.History.from_signature_pairs([
            {"question": "What is 2+2?", "answer": "4"},
        ])
        assert history.mode == "signature"
        assert len(history.messages) == 1

    def test_from_kv_creates_flat_mode(self):
        """from_kv() creates History with flat mode."""
        history = dspy.History.from_kv([
            {"thought": "I need to search", "tool": "search", "result": "Found it"},
        ])
        assert history.mode == "flat"
        assert len(history.messages) == 1


class TestHistorySerialization:
    """Tests for History serialization."""

    def test_model_dump_includes_all_fields(self):
        """model_dump() returns dict with messages and mode."""
        history = dspy.History(messages=[{"a": 1}], mode="flat")
        dumped = history.model_dump()
        assert "messages" in dumped
        assert "mode" in dumped
        assert dumped["messages"] == [{"a": 1}]
        assert dumped["mode"] == "flat"
