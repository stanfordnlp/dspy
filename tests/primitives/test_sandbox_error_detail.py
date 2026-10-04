from dspy.primitives.python_interpreter import _sandbox_error_detail


def test_syntax_error_uses_args_when_message_is_blank():
    args = ["invalid syntax", ["<exec>", 1, 12, "def broken(:\n", 1, 13]]
    detail = _sandbox_error_detail("", {"type": "SyntaxError", "args": args})
    assert isinstance(detail, str)
    assert "invalid syntax" in detail
    assert "def broken" in detail


def test_error_detail_falls_back_to_message():
    assert _sandbox_error_detail("boom", {"args": []}) == "boom"
    assert _sandbox_error_detail("boom", {}) == "boom"
    assert _sandbox_error_detail("boom", {"args": None}) == "boom"


def test_error_detail_renders_repr_fallback_args():
    assert _sandbox_error_detail("", {"args": ["<object object at 0x1>"]}) == "<object object at 0x1>"
