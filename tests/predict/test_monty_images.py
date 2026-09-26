"""Monty + sub-RLM + real isolated image editing; only LM responses are mocked."""

import asyncio
import base64
from io import BytesIO

import pytest
from PIL import Image as PILImage

import dspy
from dspy.utils.dummies import DummyLM
from tests.predict.test_rlm import make_mock_predictor

pytest.importorskip("pydantic_monty")


def decode(image):
    return PILImage.open(BytesIO(base64.b64decode(image.url.split(",", 1)[1])))


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


@pytest.mark.deno
@pytest.mark.parametrize("use_async", [False, True])
def test_nested_rlm_edits_queries_and_returns_typed_image(use_async):
    pixels = PILImage.new("RGB", (4, 3), "red")
    pixels.putpixel((2, 1), (0, 0, 255))
    image = dspy.Image(pixels)
    sessions = []

    class TrackedMonty(dspy.MontyInterpreter):
        def __init__(self):
            super().__init__()
            sessions.append(self)

    rlm = dspy.RLM(
        "source: dspy.Image -> edited: dspy.Image, answer",
        interpreter_factory=TrackedMonty,
        tools=[dspy.Image.process_images],
        max_iters=1,
        max_llm_calls=3,
    )
    rlm.generate_action = make_mock_predictor(
        [
            {
                "reasoning": "Delegate image editing, then inspect the result",
                "code": (
                    "child = dspy.RLM('source: dspy.Image -> edited: dspy.Image', tools=[process_images], max_iters=1)\n"
                    "edited = child(source=source).edited\n"
                    "description = dspy.Predict('image: dspy.Image -> answer')(image=edited).answer\n"
                    "answer = llm_query('Check the edit', images=[edited])\n"
                    "try:\n"
                    "    llm_query('Over budget', images=[edited])\n"
                    "except Exception as error:\n"
                    "    answer = answer + ':' + str(error)\n"
                    "SUBMIT(edited=edited, answer=description + ':' + answer)"
                ),
            }
        ]
    )
    edit = (
        "crop = images[0].to_pil().crop((1, 0, 4, 2)).resize((6, 4), PILImage.Resampling.NEAREST)\n"
        "array = DSPyImage.from_pil(crop).to_cv2()\n"
        "rotated = cv2.rotate(array, cv2.ROTATE_90_CLOCKWISE)\n"
        "SUBMIT(DSPyImage.from_cv2(rotated))"
    )
    lm = DummyLM(
        [
            {"reasoning": "Crop and rotate", "code": f"SUBMIT(edited=process_images({edit!r}, [source]))"},
            {"answer": "blue patch"},
            {"answer": "confirmed"},
        ]
    )
    with dspy.context(lm=lm):
        result = asyncio.run(rlm.acall(source=image)) if use_async else rlm(source=image)
    edited = decode(result.edited)
    assert edited.size == (4, 6)
    assert edited.getpixel((0, 2)) == (0, 0, 255)
    assert edited.getpixel((3, 2)) == (255, 0, 0)
    assert result.answer.startswith("blue patch:") and "confirmed" in result.answer
    assert "LLM call limit exceeded: 3 + 1 > 3" in result.answer
    assert len(result.trajectory) == 1
    assert len(lm.history) == 3  # child action + typed predictor + direct multimodal query
    for call in lm.history[1:]:
        assert any(
            part.get("type") == "image_url" and part["image_url"]["url"] == result.edited.url
            for message in call["messages"]
            if isinstance(message["content"], list)
            for part in message["content"]
        )
    assert len(sessions) == 2 and all(session._ended for session in sessions)


@pytest.mark.deno
@pytest.mark.parametrize("delegate", [False, True])
def test_flex_can_edit_images_directly_or_through_monty_rlm(delegate):
    flex = dspy.Flex(
        "source: dspy.Image -> edited: dspy.Image",
        tools=[dspy.Image.process_images],
        interpreter_factory=dspy.MontyInterpreter,
    )
    edit = (
        "array = images[0].to_cv2()\n"
        "mask = np.all(array == [0, 0, 255], axis=2)\n"
        "array[mask] = [0, 255, 0]\n"
        "array = cv2.GaussianBlur(array, (3, 3), 0)\n"
        "SUBMIT(DSPyImage.from_cv2(array))"
    )
    flex._bind_code(
        "class Editor(dspy.Module):\n"
        "    def forward(self, source):\n"
        + (
            "        edit = dspy.RLM('source: dspy.Image -> edited: dspy.Image', tools=[process_images], max_iters=1)\n"
            "        return edit(source=source)"
            if delegate
            else f"        return dspy.Prediction(edited=process_images({edit!r}, [source]))"
        )
    )
    lm = DummyLM([{"reasoning": "Mask and denoise", "code": f"SUBMIT(edited=process_images({edit!r}, [source]))"}])
    pixels = PILImage.new("RGB", (5, 3), "red")
    pixels.putpixel((0, 0), (0, 0, 255))
    with dspy.context(lm=lm):
        result = flex(source=dspy.Image(pixels))
    edited = decode(result.edited)
    assert edited.size == (5, 3)
    assert edited.getpixel((4, 2)) == (0, 255, 0)
    assert edited.getpixel((0, 0)) == (0, 191, 64)
    assert len(lm.history) == int(delegate)


@pytest.mark.deno
def test_image_worker_isolation_cleanup_and_missing_submit(monkeypatch, tmp_path):
    workers = []
    processes = []

    class TrackedWorker(dspy.PythonInterpreter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            workers.append(self)

        def shutdown(self):
            if self.deno_process is not None:
                processes.append(self.deno_process)
            super().shutdown()

    monkeypatch.setattr(dspy.primitives.python_interpreter, "PythonInterpreter", TrackedWorker)
    secret = tmp_path / "host-only.txt"
    secret.write_text("not available to image code")
    with pytest.raises(dspy.CodeExecutionError):
        dspy.Image.process_images(f"SUBMIT(open({str(secret)!r}).read())", [])
    with pytest.raises(dspy.CodeExecutionError, match="must finish with SUBMIT"):
        dspy.Image.process_images("sentinel = 41", [])
    assert dspy.Image.process_images("SUBMIT('sentinel' in globals())", []) is False
    assert len(workers) == 3
    assert all(worker._session_ended and worker.deno_process is None for worker in workers)
    assert len(processes) == 3 and all(process.poll() is not None for process in processes)


def test_image_worker_does_not_download_untrusted_urls(monkeypatch):
    def no_worker(*args, **kwargs):
        pytest.fail("A remote reference must be rejected before creating a worker")

    monkeypatch.setattr(dspy.primitives.python_interpreter, "PythonInterpreter", no_worker)
    with pytest.raises(ValueError, match="requires embedded images"):
        dspy.Image.process_images("SUBMIT(images[0])", ["http://169.254.169.254/latest/meta-data/"])
