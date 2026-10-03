from dspy.primitives.python_interpreter import _sandbox_error_detail


def test_syntax_error_uses_args_when_message_is_blank():
    args = ["invalid syntax", ["<exec>", 1, 12, "def broken(:\n", 1, 13]]
    assert _sandbox_error_detail("", {"type": "SyntaxError", "args": args}) == args


def test_error_detail_falls_back_to_message():
    assert _sandbox_error_detail("boom", {"args": []}) == "boom"
    assert _sandbox_error_detail("boom", {}) == "boom"
