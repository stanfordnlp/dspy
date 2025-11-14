import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import pytest
from litellm import Choices, Message, ModelResponse

import dspy


def test_basic_dspy_settings():
    dspy.configure(lm=dspy.LM("openai/gpt-4o"), adapter=dspy.JSONAdapter(), callbacks=[lambda x: x])
    assert dspy.settings.lm.model == "openai/gpt-4o"
    assert isinstance(dspy.settings.adapter, dspy.JSONAdapter)
    assert len(dspy.settings.callbacks) == 1


def test_forbid_configure_call_in_child_thread():
    dspy.configure(lm=dspy.LM("openai/gpt-4o"), adapter=dspy.JSONAdapter(), callbacks=[lambda x: x])

    def worker():
        with pytest.raises(RuntimeError, match="Cannot call dspy.configure"):
            dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), callbacks=[])

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(worker)


def test_dspy_context():
    dspy.configure(lm=dspy.LM("openai/gpt-4o"), adapter=dspy.JSONAdapter(), callbacks=[lambda x: x])
    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini"), callbacks=[]):
        assert dspy.settings.lm.model == "openai/gpt-4o-mini"
        assert len(dspy.settings.callbacks) == 0

    assert dspy.settings.lm.model == "openai/gpt-4o"
    assert len(dspy.settings.callbacks) == 1


def test_dspy_context_parallel():
    dspy.configure(lm=dspy.LM("openai/gpt-4o"), adapter=dspy.JSONAdapter(), callbacks=[lambda x: x])

    def worker(i):
        with dspy.context(lm=dspy.LM("openai/gpt-4o-mini"), trace=[i], callbacks=[]):
            assert dspy.settings.lm.model == "openai/gpt-4o-mini"
            assert dspy.settings.trace == [i]
            assert len(dspy.settings.callbacks) == 0

    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(worker, range(3))

    assert dspy.settings.lm.model == "openai/gpt-4o"
    assert len(dspy.settings.callbacks) == 1


def test_dspy_context_with_dspy_parallel():
    dspy.configure(lm=dspy.LM("openai/gpt-4o", cache=False), adapter=dspy.ChatAdapter())

    class MyModule(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict("question -> answer")

        def forward(self, question: str) -> str:
            lm = dspy.LM("openai/gpt-4o-mini", cache=False) if "France" in question else dspy.settings.lm
            with dspy.context(lm=lm):
                time.sleep(1)
                assert dspy.settings.lm.model == lm.model
                return self.predict(question=question)

    with mock.patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            choices=[Choices(message=Message(content="[[ ## answer ## ]]\nParis"))],
            model="openai/gpt-4o-mini",
        )

        module = MyModule()
        parallelizer = dspy.Parallel()
        input_pairs = [
            (module, {"question": "What is the capital of France?"}),
            (module, {"question": "What is the capital of Germany?"}),
        ]
        parallelizer(input_pairs)

        # Verify mock was called correctly
        assert mock_completion.call_count == 2
        for call_args in mock_completion.call_args_list:
            if "France" in call_args.kwargs["messages"][-1]["content"]:
                # France question uses gpt-4o-mini
                assert call_args.kwargs["model"] == "openai/gpt-4o-mini"
            else:
                # Germany question uses gpt-4o
                assert call_args.kwargs["model"] == "openai/gpt-4o"

        # The main thread is not affected by the context
        assert dspy.settings.lm.model == "openai/gpt-4o"


@pytest.mark.asyncio
async def test_dspy_context_with_async_task_group():
    class MyModule(dspy.Module):
        def __init__(self):
            self.predict = dspy.Predict("question -> answer")

        async def aforward(self, question: str) -> str:
            lm = (
                dspy.LM("openai/gpt-4o-mini", cache=False)
                if "France" in question
                else dspy.LM("openai/gpt-4o", cache=False)
            )
            with dspy.context(lm=lm, trace=[]):
                await asyncio.sleep(1)
                assert dspy.settings.lm.model == lm.model
                result = await self.predict.acall(question=question)
                assert len(dspy.settings.trace) == 1
                return result

    module = MyModule()

    with dspy.context(lm=dspy.LM("openai/gpt-4.1", cache=False), adapter=dspy.ChatAdapter()):
        with mock.patch("litellm.acompletion") as mock_completion:
            mock_completion.return_value = ModelResponse(
                choices=[Choices(message=Message(content="[[ ## answer ## ]]\nParis"))],
                model="openai/gpt-4o-mini",
            )

            # Define the coroutines to be run
            coroutines = [
                module.acall(question="What is the capital of France?"),
                module.acall(question="What is the capital of France?"),
                module.acall(question="What is the capital of Germany?"),
                module.acall(question="What is the capital of Germany?"),
            ]

            # Run them concurrently and gather results
            results = await asyncio.gather(*coroutines)

        assert results[0].answer == "Paris"
        assert results[1].answer == "Paris"
        assert results[2].answer == "Paris"
        assert results[3].answer == "Paris"

        # Verify mock was called correctly
        assert mock_completion.call_count == 4
        # France question uses gpt-4o-mini
        assert mock_completion.call_args_list[0].kwargs["model"] == "openai/gpt-4o-mini"
        assert mock_completion.call_args_list[1].kwargs["model"] == "openai/gpt-4o-mini"
        # Germany question uses gpt-4o
        assert mock_completion.call_args_list[2].kwargs["model"] == "openai/gpt-4o"
        assert mock_completion.call_args_list[3].kwargs["model"] == "openai/gpt-4o"

        # The main thread is not affected by the context
        assert dspy.settings.lm.model == "openai/gpt-4.1"
        assert dspy.settings.trace == []


@pytest.mark.asyncio
async def test_dspy_configure_allowance_async():
    def bar1():
        # `dspy.configure` is disallowed in different async tasks from the initial one.
        # In this case, foo1 (async) calls bar1 (sync), and bar1 uses the async task from foo1.
        with pytest.raises(RuntimeError) as e:
            dspy.configure(lm=dspy.LM("openai/gpt-4o"))
        assert "dspy.configure(...) can only be called from the same async" in str(e.value)

    async def foo1():
        bar1()
        await asyncio.sleep(0.1)

    async def foo2():
        # `dspy.configure` is disallowed in different async tasks from the initial one.
        with pytest.raises(RuntimeError) as e:
            dspy.configure(lm=dspy.LM("openai/gpt-4o"))
        assert "dspy.configure(...) can only be called from the same async" in str(e.value)
        await asyncio.sleep(0.1)

    async def foo3():
        # `dspy.context` is allowed in different async tasks from the initial one.
        with dspy.context(lm=dspy.LM("openai/gpt-4o")):
            await asyncio.sleep(0.1)

    async def foo4():
        # foo4 is directly invoked by the entry task, so it has the same async task as the entry task.
        dspy.configure(lm=dspy.LM("openai/gpt-4o"))
        await asyncio.sleep(0.1)

    # `dspy.configure` is allowed to be called multiple times in the same async task.
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
    dspy.configure(lm=dspy.LM("openai/gpt-4o"))
    dspy.configure(adapter=dspy.JSONAdapter())

    await asyncio.gather(foo1(), foo2(), foo3())

    foo4()
