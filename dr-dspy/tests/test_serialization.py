from __future__ import annotations

import pydantic

import dspy
from dr_dspy.serialization import PAYLOAD_MAX_BYTES, to_jsonable


class Solve(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ExampleModel(pydantic.BaseModel):
    x: int
    y: str


def test_to_jsonable_robustness() -> None:
    example = dspy.Example(question="why?", answer="because").with_inputs(
        "question"
    )
    assert isinstance(to_jsonable(example), dict)
    assert isinstance(to_jsonable(dspy.Prediction(code="x")), dict)

    signature_json = to_jsonable(Solve)
    assert isinstance(signature_json, dict)
    assert "signature" in signature_json

    lm_json = to_jsonable(dspy.LM("openai/gpt-4o-mini"))
    assert isinstance(lm_json, dict)
    assert lm_json.get("model") == "openai/gpt-4o-mini"

    async def coro() -> int:
        return 1

    coroutine = coro()
    try:
        coroutine_json = to_jsonable(coroutine)
        assert isinstance(coroutine_json, str)
        assert "coroutine" in coroutine_json
    finally:
        coroutine.close()

    assert to_jsonable(ExampleModel(x=1, y="a")) == {"x": 1, "y": "a"}

    weird = {
        "s": {1, 2, 3},
        "fs": frozenset({4, 5}),
        "b": b"hello",
        "nested": [{"k": (1, 2)}],
    }
    weird_json = to_jsonable(weird)
    assert isinstance(weird_json, dict)
    assert "s" in weird_json
    assert "b" in weird_json

    big = "x" * (PAYLOAD_MAX_BYTES + 1000)
    big_json = to_jsonable(big)
    assert isinstance(big_json, dict)
    assert big_json.get("_truncated") is True
