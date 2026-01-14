import pydantic
import pytest

import dspy
from dspy.utils.dummies import DummyLM


def _image_parts(messages):
    parts = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                parts.append(part)
    return parts


def _setup_predictor(signature, expected_output):
    lm = DummyLM([expected_output])
    dspy.configure(lm=lm)
    return dspy.Predict(signature), lm


def test_image_format_includes_detail_when_set():
    img = dspy.Image("https://example.com/chart.png", detail="high")
    formatted = img.format()
    assert len(formatted) == 1
    assert formatted[0]["type"] == "image_url"
    assert formatted[0]["image_url"]["url"] == "https://example.com/chart.png"
    assert formatted[0]["image_url"]["detail"] == "high"


def test_image_format_excludes_detail_by_default():
    img = dspy.Image("https://example.com/icon.png")
    formatted = img.format()
    assert len(formatted) == 1
    assert formatted[0]["type"] == "image_url"
    assert formatted[0]["image_url"]["url"] == "https://example.com/icon.png"
    assert "detail" not in formatted[0]["image_url"]


def test_image_detail_validation_rejects_invalid_values():
    with pytest.raises(pydantic.ValidationError):
        dspy.Image("https://example.com/x.png", detail="invalid")


def test_image_detail_is_emitted_in_predictor_messages_when_set():
    predictor, lm = _setup_predictor("image: dspy.Image -> answer: str", {"answer": "ok"})
    _ = predictor(image=dspy.Image("https://example.com/chart.png", detail="high"))

    parts = _image_parts(lm.history[-1]["messages"])
    assert len(parts) == 1
    assert parts[0]["image_url"]["url"] == "https://example.com/chart.png"
    assert parts[0]["image_url"]["detail"] == "high"


def test_image_detail_is_absent_in_predictor_messages_by_default():
    predictor, lm = _setup_predictor("image: dspy.Image -> answer: str", {"answer": "ok"})
    _ = predictor(image=dspy.Image("https://example.com/icon.png"))

    parts = _image_parts(lm.history[-1]["messages"])
    assert len(parts) == 1
    assert parts[0]["image_url"]["url"] == "https://example.com/icon.png"
    assert "detail" not in parts[0]["image_url"]
