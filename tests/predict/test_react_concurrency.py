import asyncio

import pytest

import dspy


def _lm(calls):
    return dspy.utils.DummyLM(
        [
            {"next_thought": "Use the tools.", "tool_calls": dspy.ToolCalls.from_dict_list(calls)},
            {
                "next_thought": "Done.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [
                        {"name": "submit", "args": {"answer": "done"}},
                    ]
                ),
            },
        ]
    )


@pytest.mark.asyncio
async def test_react_executes_independent_tools_concurrently_in_requested_order():
    started = set()
    completed = []
    both_started = asyncio.Event()

    async def lookup(query: str) -> str:
        started.add(query)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        if query == "first":
            while not completed:
                await asyncio.sleep(0)
        completed.append(query)
        return f"found {query}"

    lm = dspy.utils.DummyLM(
        [
            {
                "next_thought": "Look up both.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [
                        {"name": "lookup", "args": {"query": "first"}},
                        {"name": "lookup", "args": {"query": "second"}},
                    ]
                ),
            },
            {
                "next_thought": "Done.",
                "tool_calls": dspy.ToolCalls.from_dict_list(
                    [
                        {"name": "submit", "args": {"answer": "both"}},
                    ]
                ),
            },
        ]
    )
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = await dspy.ReAct("question -> answer", tools=[lookup], max_tool_concurrency=2).acall(
            question="Compare the two."
        )

    assert result.answer == "both"
    assert completed == ["second", "first"]
    results = result.history.messages[0]["tool_calls"].tool_call_results.tool_call_results
    assert [item.value for item in results] == ["found first", "found second"]
    assert [item.call_id for item in results] == ["call_0_0", "call_0_1"]


@pytest.mark.parametrize("limit", [0, -1, True, None, 1.5, "2"])
def test_react_rejects_invalid_tool_concurrency(limit):
    with pytest.raises(ValueError, match="max_tool_concurrency must be a positive integer"):
        dspy.ReAct("question -> answer", tools=[], max_tool_concurrency=limit)


@pytest.mark.asyncio
async def test_react_bounds_concurrency_and_keeps_tool_errors_as_observations():
    active = 0
    peak = 0
    started = []
    two_started = asyncio.Event()
    release = asyncio.Event()

    async def lookup(query: str) -> str:
        nonlocal active, peak
        started.append(query)
        active += 1
        peak = max(peak, active)
        try:
            if active == 2:
                two_started.set()
            await release.wait()
            if query == "broken":
                raise ValueError("lookup unavailable")
            return query
        finally:
            active -= 1

    calls = [{"name": "lookup", "args": {"query": query}} for query in ("first", "broken", "third")]
    calls.insert(1, {"name": "unknown", "args": {}})
    with dspy.context(lm=_lm(calls), adapter=dspy.ChatAdapter()):
        agent = dspy.ReAct("question -> answer", tools=[lookup], max_tool_concurrency=2)
        task = asyncio.create_task(agent.acall(question="test"))
        try:
            await asyncio.wait_for(two_started.wait(), timeout=1)
            assert started == ["first", "broken"]
        finally:
            release.set()
        result = await asyncio.wait_for(task, timeout=1)

    assert peak == 2
    assert active == 0
    results = result.history.messages[0]["tool_calls"].tool_call_results.tool_call_results
    assert [item.is_error for item in results] == [False, True, True, False]
    assert [results[0].value, results[3].value] == ["first", "third"]
    assert "Unknown tool" in results[1].value
    assert "lookup unavailable" in results[2].value


@pytest.mark.asyncio
async def test_react_cancellation_awaits_active_and_queued_tools():
    started = set()
    cancelled = set()
    two_started = asyncio.Event()
    never = asyncio.Event()

    async def lookup(query: str) -> str:
        started.add(query)
        if len(started) == 2:
            two_started.set()
        try:
            await never.wait()
        finally:
            await asyncio.sleep(0)
            cancelled.add(query)
        return query

    calls = [{"name": "lookup", "args": {"query": query}} for query in ("first", "second", "queued")]
    with dspy.context(lm=_lm(calls), adapter=dspy.ChatAdapter()):
        agent = dspy.ReAct("question -> answer", tools=[lookup], max_tool_concurrency=2)
        task = asyncio.create_task(agent.acall(question="test"))
        try:
            await asyncio.wait_for(two_started.wait(), timeout=1)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    assert started == cancelled == {"first", "second"}


@pytest.mark.asyncio
async def test_react_submit_batch_preserves_sequential_execution():
    steps = []

    async def lookup(query: str) -> str:
        steps.append(f"start:{query}")
        await asyncio.sleep(0)
        steps.append(f"end:{query}")
        return query

    calls = [
        {"name": "lookup", "args": {"query": "first"}},
        {"name": "submit", "args": {"answer": "done"}},
        {"name": "lookup", "args": {"query": "second"}},
    ]
    with dspy.context(lm=_lm(calls), adapter=dspy.ChatAdapter()):
        result = await dspy.ReAct("question -> answer", tools=[lookup], max_tool_concurrency=3).acall(question="test")
    assert result.answer == "done"
    assert steps == ["start:first", "end:first", "start:second", "end:second"]


@pytest.mark.asyncio
async def test_react_saved_before_concurrency_option_still_executes_sequentially():
    steps = []

    async def lookup(query: str) -> str:
        steps.append(f"start:{query}")
        await asyncio.sleep(0)
        steps.append(f"end:{query}")
        return query

    calls = [{"name": "lookup", "args": {"query": query}} for query in ("first", "second")]
    agent = dspy.ReAct("question -> answer", tools=[lookup])
    del agent.max_tool_concurrency
    with dspy.context(lm=_lm(calls), adapter=dspy.ChatAdapter()):
        result = await agent.acall(question="test")
    assert result.answer == "done"
    assert steps == ["start:first", "end:first", "start:second", "end:second"]


@pytest.mark.asyncio
async def test_react_forced_submit_with_concurrency_does_not_execute_other_tools():
    async def forbidden() -> str:
        pytest.fail("Forced submission must not execute ordinary tools")

    calls = [
        {"name": "forbidden", "args": {}},
        {"name": "submit", "args": {"answer": "done"}},
    ]
    with dspy.context(lm=_lm(calls), adapter=dspy.ChatAdapter()):
        result = await dspy.ReAct("question -> answer", tools=[forbidden], max_iters=0, max_tool_concurrency=3).acall(
            question="test"
        )
    assert result.answer == "done"
    assert result.termination_reason == "forced_submit"


@pytest.mark.asyncio
async def test_react_concurrent_batch_accepts_synchronous_tools():
    def local() -> str:
        return "local result"

    async def remote() -> str:
        await asyncio.sleep(0)
        return "remote result"

    calls = [{"name": name, "args": {}} for name in ("remote", "local", "remote")]
    with dspy.context(lm=_lm(calls), adapter=dspy.ChatAdapter()):
        result = await dspy.ReAct("question -> answer", tools=[local, remote], max_tool_concurrency=2).acall(
            question="test"
        )
    results = result.history.messages[0]["tool_calls"].tool_call_results.tool_call_results
    assert [item.value for item in results] == ["remote result", "local result", "remote result"]


@pytest.mark.asyncio
async def test_react_fatal_batch_error_awaits_sibling_cleanup():
    class StopBatch(BaseException):
        pass

    parked = asyncio.Event()
    cleaned = asyncio.Event()

    async def slow() -> str:
        parked.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cleaned.set()
        return "unreachable"

    async def abort() -> str:
        await parked.wait()
        raise StopBatch

    calls = [{"name": name, "args": {}} for name in ("slow", "abort")]
    with dspy.context(lm=_lm(calls), adapter=dspy.ChatAdapter()):
        with pytest.raises(StopBatch):
            await asyncio.wait_for(
                dspy.ReAct("question -> answer", tools=[slow, abort], max_tool_concurrency=2).acall(question="test"),
                timeout=1,
            )
    assert cleaned.is_set()
