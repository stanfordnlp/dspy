"""Offline public-flow tests: no provider credentials or network are needed."""

import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import litellm
import pytest
from openai import OpenAI

import dspy
from dspy.clients.cache import Cache
from dspy.clients.openai import OpenAIProvider
from dspy.utils.exceptions import LMProviderError, LMRateLimitError, LMTimeoutError, LMUnsupportedFeatureError


class FakeOpenAI:
    def __init__(self):
        self.inputs = {}
        self.outputs = {}
        self.jobs = {}
        self.options = []
        self.deleted = []
        self.cancelled = []
        self.closed = 0
        self.mode = "completed"
        self.submitted = threading.Event()
        self.transform = lambda rows: rows
        self.files = SimpleNamespace(create=self.upload, content=self.content, delete=self.deleted.append)
        self.batches = SimpleNamespace(create=self.create, retrieve=self.retrieve, cancel=self.cancelled.append)

    def client(self, **options):
        self.options.append(options)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed += 1

    def upload(self, *, file, purpose):
        assert purpose == "batch"
        file_id = f"input-{len(self.inputs)}"
        self.inputs[file_id] = [json.loads(line) for line in file[1].decode().splitlines()]
        return SimpleNamespace(id=file_id)

    def create(self, *, input_file_id, endpoint, completion_window):
        assert endpoint == "/v1/chat/completions"
        assert completion_window == "24h"
        self.submitted.set()
        if self.mode == "submit-error":
            raise RuntimeError("lost submission response")
        job_id = f"job-{len(self.jobs)}"
        rows = []
        for entry in reversed(self.inputs[input_file_id]):
            assert entry["url"] == endpoint and entry["method"] == "POST"
            body = entry["body"]
            content = body["messages"][-1]["content"]
            rows.append(
                {
                    "custom_id": entry["custom_id"],
                    "response": {
                        "status_code": 200,
                        "request_id": f"request-{entry['custom_id']}",
                        "body": {
                            "id": "chatcmpl-test",
                            "object": "chat.completion",
                            "created": 0,
                            "model": body["model"],
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": content},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                        },
                    },
                    "error": None,
                }
            )
        rows = self.transform(rows)
        output = f"output-{job_id}"
        errors = f"errors-{job_id}"
        self.outputs[output] = [row for row in rows if not row.get("error")]
        self.outputs[errors] = [row for row in rows if row.get("error")]
        status = "in_progress" if self.mode in {"stall", "poll-error", "poll"} else self.mode
        job = SimpleNamespace(
            id=job_id, status=status, output_file_id=output, error_file_id=errors, errors="invalid input"
        )
        self.jobs[job_id] = job
        return job

    def retrieve(self, job_id):
        if self.mode == "poll-error":
            raise RuntimeError("poll failed")
        job = self.jobs[job_id]
        if self.mode == "poll":
            job.status = "completed"
        return job

    def content(self, file_id):
        return SimpleNamespace(text="\n".join(json.dumps(row) for row in self.outputs[file_id]))


@pytest.fixture
def backend(monkeypatch, tmp_path):
    fake = FakeOpenAI()
    monkeypatch.setattr("dspy.clients.openai.openai.OpenAI", fake.client)
    monkeypatch.setattr(
        dspy, "cache", Cache(enable_disk_cache=False, enable_memory_cache=True, disk_cache_dir=tmp_path)
    )

    def forbid_online(*args, **kwargs):
        pytest.fail("Provider batch must not call the synchronous completion API")

    monkeypatch.setattr(litellm, "completion", forbid_online)
    yield fake
    assert not any(thread.name == "dspy-batch-dispatch" for thread in threading.enumerate())
    assert dspy.settings.get("batch_coordinator") is None
    assert fake.closed == len(fake.options)


class Program(dspy.Module):
    def forward(self, value, steps=1, fail=None, **config):
        if fail == "before":
            raise ValueError("local failure before LM")
        answers = []
        for step in range(steps):
            answers.append(dspy.settings.lm(f"{value}:{step}", **config)[0])
            if fail == "after":
                raise ValueError("local failure after LM")
        return dspy.Prediction(answers=answers)


def examples(*items):
    return [dspy.Example(**item).with_inputs(*item) for item in items]


def run(program, data, **kwargs):
    lm = dspy.LM("openai/gpt-4o-mini", cache=False)
    with dspy.context(lm=lm):
        return program.batch(data, batch_mode=True, disable_progress_bar=True, batch_poll_interval=0.001, **kwargs)


@pytest.mark.parametrize("steps", [(2, 2), (1, 3), (0, 2)])
@pytest.mark.parametrize("num_threads", [1, 2])
def test_equal_unequal_and_zero_call_programs(backend, steps, num_threads):
    result = run(
        Program(),
        examples({"value": "a", "steps": steps[0]}, {"value": "b", "steps": steps[1]}),
        num_threads=num_threads,
    )
    assert [r.answers for r in result] == [
        [f"{value}:{i}" for i in range(count)] for value, count in zip("ab", steps, strict=False)
    ]
    assert sum(len(entries) for entries in backend.inputs.values()) == sum(steps)
    assert set(backend.deleted) == set(backend.inputs) | set(backend.outputs)


def test_empty(backend):
    assert run(Program(), []) == []
    assert not backend.inputs


@pytest.mark.parametrize("fail", ["before", "after"])
def test_local_errors_keep_successes_and_failure_details(backend, fail):
    result, failed, errors = run(
        Program(),
        examples({"value": "a", "fail": fail}, {"value": "b", "steps": 2}),
        return_failed_examples=True,
        max_errors=5,
        num_threads=2,
    )
    assert result[0] is None and result[1].answers == ["b:0", "b:1"]
    assert len(failed) == len(errors) == 1
    assert failed[0].value == "a" and isinstance(errors[0], ValueError)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unknown", "malformed", "error", "status"])
def test_invalid_results_resolve_waiters(backend, problem):
    def transform(rows):
        if problem == "missing":
            return []
        if problem == "duplicate":
            return rows + rows
        if problem == "unknown":
            rows[0]["custom_id"] = "unknown"
        if problem == "malformed":
            rows[0]["response"]["body"] = {}
        if problem == "error":
            for row in rows:
                row.update(response=None, error={"message": "request expired", "code": "batch_expired"})
        if problem == "status":
            for row in rows:
                row["response"].update(status_code=429, body={"error": {"message": "quota", "code": "rate_limit"}})
        return rows

    backend.transform = transform
    result, failed, errors = run(Program(), examples({"value": "a"}), return_failed_examples=True, max_errors=5)
    assert result == [None] and len(failed) == len(errors) == 1
    assert isinstance(errors[0], dspy.LMError)
    if problem == "status":
        assert isinstance(errors[0], LMRateLimitError)
        assert errors[0].status == 429


@pytest.mark.parametrize("mode", ["failed", "submit-error", "poll-error", "unexpected-status"])
def test_batch_level_failure(backend, mode):
    backend.mode = mode
    result, failed, errors = run(
        Program(), examples({"value": "a"}, {"value": "b"}), return_failed_examples=True, num_threads=2, max_errors=5
    )
    assert result == [None, None] and len(failed) == len(errors) == 2
    assert all(isinstance(error, dspy.LMError) for error in errors)
    if mode == "submit-error":
        assert not backend.deleted  # Outcome unknown: don't destroy recovery evidence.
    if mode in {"poll-error", "unexpected-status"}:
        assert set(backend.cancelled) == set(backend.jobs)


@pytest.mark.parametrize("mode", ["poll", "expired", "cancelled"])
def test_polling_and_partial_terminal_results(backend, mode):
    backend.mode = mode
    result = run(Program(), examples({"value": "a"}))
    assert result[0].answers == ["a:0"]
    assert not backend.cancelled


def test_timeout_cancels_remote_job_without_resubmission(backend):
    backend.mode = "stall"
    _, _, errors = run(
        Program(), examples({"value": "a"}), return_failed_examples=True, batch_timeout=0.02, timeout=0.001
    )
    assert isinstance(errors[0], LMTimeoutError)
    assert len(backend.jobs) == 1 and backend.cancelled == ["job-0"]
    assert not backend.deleted


def test_max_errors_cancels_other_waiting_workers(backend):
    backend.mode = "stall"

    class FailingProgram(Program):
        def forward(self, value):
            if value == "fail":
                assert backend.submitted.wait(2)
                raise ValueError("stop")
            return super().forward(value)

    with pytest.raises(Exception, match="cancelled"):
        run(FailingProgram(), examples({"value": "wait"}, {"value": "fail"}), num_threads=2, max_errors=1)
    assert backend.cancelled == ["job-0"]


def test_effective_kwargs_and_client_configuration(backend):
    lm = dspy.LM("openai/gpt-4o-mini", cache=False, temperature=0.4, max_tokens=100, api_key="fake-default")
    with dspy.context(lm=lm):
        result = Program().batch(
            examples(
                {
                    "value": "a",
                    "temperature": 0.8,
                    "max_tokens": 23,
                    "api_key": "fake-override",
                    "api_base": "https://example.invalid/v1",
                    "headers": {"X-Test": "test"},
                    "rollout_id": 2,
                    "response_format": {"type": "json_object"},
                }
            ),
            batch_mode=True,
            disable_progress_bar=True,
        )
    assert result[0].answers == ["a:0"]
    body = next(iter(backend.inputs.values()))[0]["body"]
    assert body == {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "a:0"}],
        "temperature": 0.8,
        "max_tokens": 23,
        "response_format": {"type": "json_object"},
    }
    assert backend.options[0]["api_key"] == "fake-override"
    assert backend.options[0]["base_url"] == "https://example.invalid/v1"
    assert backend.options[0]["default_headers"] == {"X-Test": "test"}
    assert backend.options[0]["max_retries"] == 0


def test_typed_request_cache_usage_history_and_settings(backend):
    lm = dspy.LM("openai/gpt-4o-mini")

    class TypedProgram(dspy.Module):
        def forward(self, value):
            response = dspy.settings.lm(dspy.LMRequest.from_call(model=lm.model, prompt=value))
            assert isinstance(response, dspy.LMResponse)
            assert dspy.settings.test_batch_setting == "inherited"
            return dspy.Prediction(answer=response.to_legacy_outputs()[0])

    with dspy.context(lm=lm, test_batch_setting="inherited", track_usage=True):
        for iteration in range(2):
            result = TypedProgram().batch(examples({"value": "typed"}), batch_mode=True, disable_progress_bar=True)
            assert result[0].answer == "typed"
            usage = result[0].get_lm_usage()
            assert sum(value["total_tokens"] for value in usage.values()) == (3 if iteration == 0 else 0)
    assert len(backend.jobs) == 1
    assert len(lm.history) == 2


def test_heterogeneous_parallel_and_lms(backend):
    class First(dspy.Module):
        def forward(self, value):
            return dspy.settings.lm(value)[0]

    class Second(dspy.Module):
        def forward(self, value):
            return dspy.LM("openai/gpt-4o", cache=False)(value + "!")[0]

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False)):
        result = dspy.Parallel(2, batch_mode=True, disable_progress_bar=True)(
            [(First(), {"value": "first"}), (Second(), {"value": "second"})]
        )
    assert result == ["first", "second!"]
    assert {entry["body"]["model"] for entries in backend.inputs.values() for entry in entries} == {
        "gpt-4o",
        "gpt-4o-mini",
    }


@pytest.mark.parametrize("model_type", ["text", "responses"])
def test_unsupported_model_types_do_not_submit(backend, model_type):
    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", model_type=model_type)):
        _, _, errors = Program().batch(
            examples({"value": "a"}), batch_mode=True, return_failed_examples=True, disable_progress_bar=True
        )
    assert isinstance(errors[0], LMUnsupportedFeatureError)
    assert not backend.inputs


def test_unsupported_provider_does_not_submit(backend):
    with dspy.context(lm=dspy.LM("anthropic/claude-sonnet-4-5", cache=False)):
        _, _, errors = Program().batch(
            examples({"value": "a"}), batch_mode=True, return_failed_examples=True, disable_progress_bar=True
        )
    assert isinstance(errors[0], LMUnsupportedFeatureError)
    assert not backend.inputs


@pytest.mark.asyncio
async def test_sync_batch_inside_event_loop(backend):
    assert run(Program(), examples({"value": "a"}), num_threads=1)[0].answers == ["a:0"]


def test_async_lm_is_explicitly_rejected(backend):
    class AsyncProgram(dspy.Module):
        def forward(self, value):
            return asyncio.run(dspy.settings.lm.acall(value))

    _, _, errors = run(AsyncProgram(), examples({"value": "a"}), return_failed_examples=True)
    assert isinstance(errors[0], LMUnsupportedFeatureError)
    assert not backend.inputs


def test_existing_positional_apis_do_not_enable_batch(backend, monkeypatch):
    def online_completion(**kwargs):
        return litellm.ModelResponse(
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": kwargs["messages"][-1]["content"]},
                    "finish_reason": "stop",
                }
            ]
        )

    monkeypatch.setattr(litellm, "completion", online_completion)

    class RegularLMProgram(dspy.Module):
        def forward(self, value):
            return dspy.settings.lm(str(value))[0]

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False)):
        assert RegularLMProgram().batch(examples({"value": 1}), 2, 3, False, False, True, 0, 3) == ["1"]
        assert dspy.Parallel(2, 3, True, False, False, True, 0, 3)([(RegularLMProgram(), {"value": 2})]) == ["2"]
    assert not backend.inputs


def test_provider_rejects_wrong_cardinality(backend):
    class BrokenProvider(OpenAIProvider):
        def batch(self, *args, **kwargs):
            return []

    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", provider=BrokenProvider(), cache=False)):
        _, _, errors = Program().batch(
            examples({"value": "a"}), batch_mode=True, return_failed_examples=True, disable_progress_bar=True
        )
    assert isinstance(errors[0], LMProviderError)


@pytest.mark.parametrize("experimental", [False, True])
@pytest.mark.parametrize("json_adapter", [False, True])
def test_predict_data_dependent_calls(backend, experimental, json_adapter):
    def answer(rows):
        for row in rows:
            message = row["response"]["body"]["choices"][0]["message"]
            value = "continue" if "\nlong\n" in message["content"] else "stop"
            message["content"] = (
                json.dumps({"answer": value})
                if json_adapter
                else f"[[ ## answer ## ]]\n{value}\n\n[[ ## completed ## ]]"
            )
        return rows

    backend.transform = answer

    class PredictProgram(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.Predict("value -> answer")

        def forward(self, value):
            result = self.predict(value=value)
            if result.answer == "continue":
                return self.predict(value=result.answer)
            return result

    adapter = dspy.JSONAdapter() if json_adapter else dspy.ChatAdapter()
    with dspy.context(experimental=experimental, adapter=adapter):
        result = run(PredictProgram(), examples({"value": "short"}, {"value": "long"}), num_threads=2)
    assert [r.answer for r in result] == ["stop", "stop"]
    assert sum(len(entries) for entries in backend.inputs.values()) == 3
    if json_adapter:
        formats = [entry["body"]["response_format"] for entries in backend.inputs.values() for entry in entries]
        assert all(format["type"] == "json_schema" and format["json_schema"]["strict"] for format in formats)


def test_keyboard_interrupt_cancels_and_closes(backend, monkeypatch):
    backend.mode = "stall"

    def interrupt(*args, **kwargs):
        assert backend.submitted.wait(2)
        raise KeyboardInterrupt

    monkeypatch.setattr("dspy.utils.parallelizer.wait", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(Program(), examples({"value": "a"}), num_threads=2)
    assert backend.cancelled == ["job-0"]


def test_ready_calls_are_coalesced_and_partitioned_by_credentials(backend):
    # The provider boundary also accepts a cohort deterministically, so this
    # assertion does not rely on the OS scheduling all workers in 10 ms.
    lm = dspy.LM("openai/gpt-4o-mini", cache=False)
    requests = [
        {"model": lm.model, "messages": [{"role": "user", "content": str(i)}], "api_key": key}
        for i, key in enumerate(["fake-a", "fake-b", "fake-a"])
    ]
    result = lm.provider.batch(lm, requests, cancel_event=threading.Event(), timeout=1, poll_interval=0.001)
    assert [r.choices[0].message.content for r in result] == ["0", "1", "2"]
    assert sorted(len(entries) for entries in backend.inputs.values()) == [1, 2]
    assert {options["api_key"] for options in backend.options} == {"fake-a", "fake-b"}


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_fails_before_submission(backend, value):
    with pytest.raises(ValueError, match="finite and positive"):
        run(Program(), [], batch_timeout=value)
    assert not backend.inputs


def test_public_openai_sdk_with_offline_http_transport(monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        if request.method == "POST" and request.url.path == "/v1/files":
            assert b'"model": "gpt-4o-mini"' in request.content
            assert b"fake-sdk-key" not in request.content
            return httpx.Response(
                200,
                json={
                    "id": "input",
                    "object": "file",
                    "bytes": 1,
                    "created_at": 0,
                    "filename": "input.jsonl",
                    "purpose": "batch",
                },
            )
        if request.method == "POST" and request.url.path == "/v1/batches":
            assert json.loads(request.content) == {
                "input_file_id": "input",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            }
            return httpx.Response(
                200,
                json={
                    "id": "job",
                    "object": "batch",
                    "status": "completed",
                    "endpoint": "/v1/chat/completions",
                    "completion_window": "24h",
                    "input_file_id": "input",
                    "output_file_id": "output",
                    "created_at": 0,
                },
            )
        if request.method == "GET" and request.url.path == "/v1/files/output/content":
            return httpx.Response(
                200,
                text=json.dumps(
                    {
                        "custom_id": "0",
                        "response": {
                            "status_code": 200,
                            "body": {
                                "id": "chatcmpl-sdk",
                                "model": "gpt-4o-mini",
                                "object": "chat.completion",
                                "created": 0,
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {"role": "assistant", "content": "sdk works"},
                                        "finish_reason": "stop",
                                    }
                                ],
                            },
                        },
                    }
                ),
            )
        if request.method == "DELETE" and request.url.path in {"/v1/files/input", "/v1/files/output"}:
            return httpx.Response(
                200, json={"id": request.url.path.rsplit("/", 1)[1], "object": "file", "deleted": True}
            )
        raise AssertionError(f"Unexpected HTTP request: {request.method} {request.url.path}")

    def client(**options):
        return OpenAI(**options, http_client=httpx.Client(transport=httpx.MockTransport(handle)))

    monkeypatch.setattr("dspy.clients.openai.openai.OpenAI", client)
    with dspy.context(lm=dspy.LM("openai/gpt-4o-mini", cache=False, api_key="fake-sdk-key")):
        result = Program().batch(examples({"value": "a"}), batch_mode=True, disable_progress_bar=True)
    assert result[0].answers == ["sdk works"]
    assert len(requests) == 5
