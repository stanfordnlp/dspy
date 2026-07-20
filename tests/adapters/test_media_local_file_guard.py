"""Regression tests for #10067 — LM output must not read local files."""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import dspy
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types.image import encode_image


def test_encode_image_does_not_read_local_file_by_default(tmp_path: Path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP-SECRET-TOKEN")
    # Without allow_local_files, a path-looking string is rejected rather than read.
    result = encode_image(str(secret))
    # Path is left as-is; contents are never read into a data URI.
    assert result == str(secret)
    assert "data:" not in result


def test_encode_image_reads_local_file_when_opted_in(tmp_path: Path):
    secret = tmp_path / "photo.png"
    # Minimal valid 1x1 PNG
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    secret.write_bytes(png)
    result = encode_image(str(secret), allow_local_files=True)
    assert result.startswith("data:image/")


def test_image_from_file_still_works(tmp_path: Path):
    secret = tmp_path / "photo.png"
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    secret.write_bytes(png)
    img = dspy.Image.from_file(str(secret))
    assert img.url.startswith("data:image/")


def test_json_adapter_parse_does_not_exfiltrate_local_file(tmp_path: Path):
    secret = tmp_path / "SECRET_victim.txt"
    secret.write_text("TOP-SECRET-AWS-KEY=AKIA_DEADBEEF")

    class S(dspy.Signature):
        """Return an image."""

        query: str = dspy.InputField()
        result_image: dspy.Image = dspy.OutputField()

    completion = f'{{"result_image": {{"url": "{secret.as_posix()}"}}}}'
    parsed = JSONAdapter().parse(S, completion)
    url = parsed["result_image"].url
    # Path may be stored as a string, but contents must never be base64-encoded.
    secret_b64 = base64.b64encode(b"TOP-SECRET-AWS-KEY=AKIA_DEADBEEF").decode()
    assert secret_b64 not in url
    assert "TOP-SECRET-AWS-KEY" not in url
    assert not url.startswith("data:")
