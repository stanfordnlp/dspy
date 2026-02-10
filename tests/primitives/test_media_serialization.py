"""
Tests for media type helpers in python_interpreter.py.

These tests do NOT require Deno — they test pure Python serialization logic
for Audio and Image objects in the sandboxed interpreter.
"""


from dspy.primitives.python_interpreter import _is_media_type, _media_descriptor


class TestIsMediaType:
    """Tests for _is_media_type helper."""

    def test_audio(self):
        from dspy.adapters.types.audio import Audio

        audio = Audio(data="dGVzdA==", audio_format="wav")
        assert _is_media_type(audio) is True

    def test_image(self):
        from dspy.adapters.types.image import Image

        img = Image(url="data:image/png;base64,iVBORw0KGgo=")
        assert _is_media_type(img) is True

    def test_string(self):
        assert _is_media_type("hello") is False

    def test_int(self):
        assert _is_media_type(42) is False

    def test_none(self):
        assert _is_media_type(None) is False

    def test_dict(self):
        assert _is_media_type({"data": "abc"}) is False

    def test_list(self):
        assert _is_media_type([1, 2, 3]) is False


class TestMediaDescriptor:
    """Tests for _media_descriptor helper."""

    def test_audio_descriptor(self):
        from dspy.adapters.types.audio import Audio

        audio = Audio(data="dGVzdA==", audio_format="wav")
        desc = _media_descriptor(audio)
        assert "Audio" in desc
        assert "wav" in desc
        assert "8" in desc  # len("dGVzdA==") == 8

    def test_image_descriptor(self):
        from dspy.adapters.types.image import Image

        img = Image(url="data:image/png;base64,iVBORw0KGgo=")
        desc = _media_descriptor(img)
        assert "Image" in desc

    def test_non_media_falls_back_to_repr(self):
        desc = _media_descriptor("just a string")
        assert desc == repr("just a string")

    def test_audio_descriptor_includes_format(self):
        from dspy.adapters.types.audio import Audio

        audio = Audio(data="AAAA", audio_format="mpeg")
        desc = _media_descriptor(audio)
        assert "mpeg" in desc

    def test_audio_descriptor_includes_data_length(self):
        from dspy.adapters.types.audio import Audio

        audio = Audio(data="A" * 100, audio_format="wav")
        desc = _media_descriptor(audio)
        assert "100" in desc
