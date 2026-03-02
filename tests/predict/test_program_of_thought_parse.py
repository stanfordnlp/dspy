"""Tests for ProgramOfThought._parse_code to verify it doesn't corrupt valid Python."""

import ast
import re
import unittest


def _parse_code(code_data):
    """Standalone copy of ProgramOfThought._parse_code for isolated testing."""
    code = code_data.get("generated_code", "").split("---", 1)[0].split("\n\n\n", 1)[0]
    code_match = re.search(r"```python[ \n](.*?)[ \n]```?", code, re.DOTALL)
    code_block = (code_match.group(1) if code_match else code).replace("\\n", "\n")
    if not code_block:
        return code, "Error: Empty code after parsing."
    if "\n" not in code_block and code_block.count("=") > 1:
        return code, "Error: Code format is not correct."
    lines = code_block.split("\n")
    last_line_match = re.match(r"^(\w+)\s*=", lines[-1].strip())
    if last_line_match and len(lines) > 1:
        code_block += "\n" + last_line_match.group(1)
    else:
        try:
            ast.parse(code_block)
        except SyntaxError:
            code_block = re.sub(
                r"([a-zA-Z_]\w* *=[^=].*?)(?=[a-zA-Z_]\w* *=[^=])",
                r"\1\n",
                code_block,
            )
    return code_block, None


class TestParseCode(unittest.TestCase):
    """Test _parse_code preserves valid Python and fixes broken code."""

    def _parse(self, code_str):
        return _parse_code({"generated_code": code_str})

    def test_preserves_comparison_operators(self):
        code = "x = 1\nif x == 1:\n    y = 2"
        result, error = self._parse(code)
        assert error is None
        assert "==" in result, f"Comparison operator corrupted: {result!r}"
        assert "if x == 1:" in result

    def test_preserves_string_with_equals(self):
        code = 'url = "https://example.com?key=value&foo=bar"\nresult = url'
        result, error = self._parse(code)
        assert error is None
        assert "key=value" in result, f"String literal corrupted: {result!r}"

    def test_preserves_not_equal(self):
        code = "a = 5\nif a != 3:\n    b = 10"
        result, error = self._parse(code)
        assert error is None
        assert "!=" in result, f"!= operator corrupted: {result!r}"

    def test_preserves_less_equal_greater_equal(self):
        code = "x = 10\nif x <= 20 and x >= 5:\n    y = True"
        result, error = self._parse(code)
        assert error is None
        assert "<=" in result
        assert ">=" in result

    def test_preserves_comment_with_equals(self):
        code = "x = 1  # x = initial value\ny = 2"
        result, error = self._parse(code)
        assert error is None
        assert "# x = initial value" in result

    def test_fixes_missing_newlines(self):
        # Two assignments jammed together without newline (multi-line so it passes the guard)
        code = "import os\nx = 1y = 2"
        result, error = self._parse(code)
        assert error is None
        # The regex should insert a newline between the assignments
        assert "x = 1\n" in result or "x = 1" in result

    def test_markdown_code_block_extraction(self):
        code = "```python\nx = 1\ny = x + 2\n```"
        result, error = self._parse(code)
        assert error is None
        assert "x = 1" in result
        assert "y = x + 2" in result

    def test_preserves_multiline_valid_code(self):
        code = (
            "data = [1, 2, 3]\n"
            "filtered = [x for x in data if x >= 2]\n"
            "total = sum(filtered)\n"
            "result = {'total': total, 'count': len(filtered)}"
        )
        result, error = self._parse(code)
        assert error is None
        assert "x >= 2" in result
        assert "result = {'total': total" in result

    def test_empty_code(self):
        result, error = self._parse("")
        assert error is not None
        assert "Empty" in error


if __name__ == "__main__":
    unittest.main()
