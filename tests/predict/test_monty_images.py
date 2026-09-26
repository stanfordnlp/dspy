"""Real Monty image transport through RLM and Flex; only LM responses are mocked."""

import asyncio

import pytest

import dspy
from dspy.utils.dummies import DummyLM
from tests.predict.test_rlm import make_mock_predictor

pytest.importorskip("pydantic_monty")


@pytest.mark.parametrize("use_async", [False, True])
def test_image_queries_need_no_pyodide(use_async, monkeypatch):
    def no_worker(*args, **kwargs):
        pytest.fail("Viewing an image must not start Pyodide")

    monkeypatch.setattr(dspy.primitives.python_interpreter, "PythonInterpreter", no_worker)
    image = dspy.Image("https://example.com/image.png")
    rlm = dspy.RLM("images -> answer", interpreter_factory=dspy.MontyInterpreter, max_iters=1)
    rlm.generate_action = make_mock_predictor(
        [
            {
                "reasoning": "Inspect images",
                "code": "SUBMIT(answer=llm_query_batched(['one', 'two'], images=[[images[0]], [images[1]['image']]])[1])",
            }
        ]
    )
    lm = DummyLM([{"answer": "seen"}, {"answer": "seen"}])
    with dspy.context(lm=lm):
        inputs = {"images": [image, {"image": image}]}
        result = asyncio.run(rlm.acall(**inputs)) if use_async else rlm(**inputs)
    assert "seen" in result.answer
    assert len(lm.history) == 2
    for call in lm.history:
        assert call["messages"][0]["content"][1] == {"type": "image_url", "image_url": {"url": image.url}}


def test_typed_image_lists_round_trip_through_chain_of_thought():
    image = dspy.Image("https://example.com/image.png")
    rlm = dspy.RLM(
        "images: list[dspy.Image] -> selected: list[dspy.Image]",
        interpreter_factory=dspy.MontyInterpreter,
        max_iters=1,
    )
    rlm.generate_action = make_mock_predictor(
        [
            {
                "reasoning": "Forward typed images",
                "code": (
                    "inspect = dspy.ChainOfThought('images: list[dspy.Image] -> selected: list[dspy.Image]')\n"
                    "SUBMIT(selected=inspect(images=images).selected)"
                ),
            }
        ]
    )
    lm = DummyLM([{"reasoning": "Select", "selected": [{"url": image.url}]}])
    with dspy.context(lm=lm):
        result = rlm(images=[image])
    assert result.selected == [image]
    assert any(
        part.get("type") == "image_url" and part["image_url"]["url"] == image.url
        for message in lm.history[0]["messages"]
        if isinstance(message["content"], list)
        for part in message["content"]
    )


@pytest.mark.parametrize("use_async", [False, True])
def test_nested_rlm_queries_and_returns_typed_image(use_async):
    image = dspy.Image("https://example.com/source.png")
    sessions = []

    class TrackedMonty(dspy.MontyInterpreter):
        def __init__(self):
            super().__init__()
            sessions.append(self)

    rlm = dspy.RLM(
        "source: dspy.Image -> selected: dspy.Image, answer",
        interpreter_factory=TrackedMonty,
        max_iters=1,
        max_llm_calls=3,
    )
    rlm.generate_action = make_mock_predictor(
        [
            {
                "reasoning": "Delegate image selection, then inspect the result",
                "code": (
                    "child = dspy.RLM('source: dspy.Image -> selected: dspy.Image', max_iters=1)\n"
                    "selected = child(source=source).selected\n"
                    "description = dspy.Predict('image: dspy.Image -> answer')(image=selected).answer\n"
                    "answer = llm_query('Check the selection', images=[selected])\n"
                    "try:\n"
                    "    llm_query('Over budget', images=[selected])\n"
                    "except Exception as error:\n"
                    "    answer = answer + ':' + str(error)\n"
                    "SUBMIT(selected=selected, answer=description + ':' + answer)"
                ),
            }
        ]
    )
    lm = DummyLM(
        [
            {"reasoning": "Forward the image", "code": "SUBMIT(selected=source)"},
            {"answer": "source image"},
            {"answer": "confirmed"},
        ]
    )
    with dspy.context(lm=lm):
        result = asyncio.run(rlm.acall(source=image)) if use_async else rlm(source=image)
    assert result.selected == image
    assert result.answer.startswith("source image:") and "confirmed" in result.answer
    assert "LLM call limit exceeded: 3 + 1 > 3" in result.answer
    assert len(result.trajectory) == 1
    assert len(lm.history) == 3  # child action + typed predictor + direct multimodal query
    for call in lm.history[1:]:
        assert any(
            part.get("type") == "image_url" and part["image_url"]["url"] == image.url
            for message in call["messages"]
            if isinstance(message["content"], list)
            for part in message["content"]
        )
    assert len(sessions) == 2 and all(session._ended for session in sessions)


def test_flex_transports_image_and_returns_typed_output():
    flex = dspy.Flex(
        "source: dspy.Image -> selected: dspy.Image",
        interpreter_factory=dspy.MontyInterpreter,
    )
    flex._bind_code(
        "class Editor(dspy.Module):\n"
        "    def forward(self, source):\n"
        "        select = dspy.Predict('image: dspy.Image -> selected: dspy.Image')\n"
        "        return select(image=source)"
    )
    image = dspy.Image("https://example.com/flex.png")
    lm = DummyLM([{"selected": {"url": image.url}}])
    with dspy.context(lm=lm):
        result = flex(source=image)
    assert result.selected == image
    assert any(
        part.get("type") == "image_url" and part["image_url"]["url"] == image.url
        for message in lm.history[0]["messages"]
        if isinstance(message["content"], list)
        for part in message["content"]
    )
