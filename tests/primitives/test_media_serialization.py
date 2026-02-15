"""
Tests for RLM sandbox type protocol in python_interpreter.py and type classes.

Tests the generic _has_rlm_support() helper and the to_sandbox/rlm_preview/sandbox_setup/
sandbox_assignment protocol on Audio and Image types.

These tests do NOT require Deno — they test pure Python serialization logic.
"""


from dspy.primitives.python_interpreter import _has_rlm_support

# ============================================================================
# Tests: _has_rlm_support helper
# ============================================================================


class TestHasRlmSupport:
    """Tests for the generic _has_rlm_support() protocol check."""

    def test_audio(self):
        from dspy.adapters.types.audio import Audio
        audio = Audio(data="dGVzdA==", audio_format="wav")
        assert _has_rlm_support(audio) is True

    def test_image(self):
        from dspy.adapters.types.image import Image
        img = Image(url="data:image/png;base64,iVBORw0KGgo=")
        assert _has_rlm_support(img) is True

    def test_string(self):
        assert _has_rlm_support("hello") is False

    def test_int(self):
        assert _has_rlm_support(42) is False

    def test_none(self):
        assert _has_rlm_support(None) is False

    def test_dict(self):
        assert _has_rlm_support({"data": "abc"}) is False

    def test_list(self):
        assert _has_rlm_support([1, 2, 3]) is False


# ============================================================================
# Tests: Audio RLM sandbox protocol
# ============================================================================


class TestAudioRlmProtocol:
    """Tests for Audio.rlm_preview, to_sandbox, sandbox_setup, sandbox_assignment."""

    def test_rlm_preview_basic(self):
        from dspy.adapters.types.audio import Audio
        audio = Audio(data="dGVzdA==", audio_format="wav")
        preview = audio.rlm_preview()
        assert "Audio" in preview
        assert "wav" in preview
        assert "8" in preview  # len("dGVzdA==") == 8

    def test_rlm_preview_includes_format(self):
        from dspy.adapters.types.audio import Audio
        audio = Audio(data="AAAA", audio_format="mpeg")
        preview = audio.rlm_preview()
        assert "mpeg" in preview

    def test_rlm_preview_includes_data_length(self):
        from dspy.adapters.types.audio import Audio
        audio = Audio(data="A" * 100, audio_format="wav")
        preview = audio.rlm_preview()
        assert "100" in preview

    def test_to_sandbox_returns_bytes(self):
        from dspy.adapters.types.audio import Audio
        audio = Audio(data="dGVzdA==", audio_format="wav")
        payload = audio.to_sandbox()
        assert isinstance(payload, bytes)
        assert b"Audio" in payload
        assert b"wav" in payload

    def test_sandbox_setup_empty(self):
        from dspy.adapters.types.audio import Audio
        audio = Audio(data="dGVzdA==", audio_format="wav")
        assert audio.sandbox_setup() == ""

    def test_sandbox_assignment(self):
        from dspy.adapters.types.audio import Audio
        audio = Audio(data="dGVzdA==", audio_format="wav")
        code = audio.sandbox_assignment("my_audio", "open('/tmp/data.json').read()")
        assert "my_audio" in code
        assert "/tmp/data.json" in code


# ============================================================================
# Tests: Image RLM sandbox protocol
# ============================================================================


class TestImageRlmProtocol:
    """Tests for Image.rlm_preview, to_sandbox, sandbox_setup, sandbox_assignment."""

    def test_rlm_preview_base64(self):
        from dspy.adapters.types.image import Image
        img = Image(url="data:image/png;base64,iVBORw0KGgo=")
        preview = img.rlm_preview()
        assert "Image" in preview
        assert "png" in preview

    def test_rlm_preview_url(self):
        from dspy.adapters.types.image import Image
        img = Image(url="https://example.com/photo.jpg", download=False)
        preview = img.rlm_preview()
        assert "Image" in preview
        assert "example.com" in preview

    def test_to_sandbox_returns_bytes(self):
        from dspy.adapters.types.image import Image
        img = Image(url="data:image/png;base64,iVBORw0KGgo=")
        payload = img.to_sandbox()
        assert isinstance(payload, bytes)
        assert b"Image" in payload

    def test_sandbox_setup_empty(self):
        from dspy.adapters.types.image import Image
        img = Image(url="data:image/png;base64,iVBORw0KGgo=")
        assert img.sandbox_setup() == ""

    def test_sandbox_assignment(self):
        from dspy.adapters.types.image import Image
        img = Image(url="data:image/png;base64,iVBORw0KGgo=")
        code = img.sandbox_assignment("my_img", "open('/tmp/img.json').read()")
        assert "my_img" in code
        assert "/tmp/img.json" in code


# ============================================================================
# Tests: Custom type with RLM protocol
# ============================================================================


class TestCustomTypeRlmProtocol:
    """Tests that any object implementing the protocol is recognized."""

    def test_custom_type_with_protocol(self):
        class MyType:
            def to_sandbox(self):
                return b"custom data"
            def rlm_preview(self):
                return "<MyType: custom>"
            def sandbox_setup(self):
                return ""
            def sandbox_assignment(self, var_name, data_expr):
                return f"{var_name} = {data_expr}"

        obj = MyType()
        assert _has_rlm_support(obj) is True

    def test_custom_type_without_protocol(self):
        class PlainType:
            pass
        assert _has_rlm_support(PlainType()) is False

    def test_partial_protocol_not_enough(self):
        """Object with rlm_preview but no to_sandbox should NOT have RLM support."""
        class HalfType:
            def rlm_preview(self):
                return "preview"
        assert _has_rlm_support(HalfType()) is False
