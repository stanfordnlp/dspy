import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import numpy as np
import openai
import pytest

import dspy
from dspy.clients.embedding import Embedder


@pytest.fixture
def cache(tmp_path):
    original_cache = dspy.cache
    dspy.configure_cache(disk_cache_dir=tmp_path / ".dspy_cache")
    yield
    dspy.cache.disk_cache.close()
    dspy.cache = original_cache


@pytest.fixture
def sdk(monkeypatch):
    """Exercise real SDK serialization and routing, but never open a network connection."""
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT_ID",
        "AZURE_OPENAI_API_KEY",
        "AZURE_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_API_BASE",
        "OPENAI_API_VERSION",
        "AZURE_API_VERSION",
        "AZURE_OPENAI_AD_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    state = SimpleNamespace(requests=[], clients=[], options=[], status_code=200)

    def respond(request):
        state.requests.append(request)
        if state.status_code != 200:
            return httpx.Response(state.status_code, json={"error": {"message": "test failure"}})
        body = json.loads(request.content)
        data = []
        for index, text in enumerate(body["input"]):
            vector = [float(len(text)), 1.0, 2.0]
            if body.get("encoding_format") == "base64":
                vector = base64.b64encode(np.array(vector, dtype="<f4").tobytes()).decode()
            data.append({"object": "embedding", "index": index, "embedding": vector})
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": body["model"],
                "data": list(reversed(data)),
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    def factory(client_type, asynchronous):
        def create(**kwargs):
            state.options.append((client_type.__name__, kwargs.copy()))
            http_client = httpx.AsyncClient if asynchronous else httpx.Client
            client = client_type(http_client=http_client(transport=httpx.MockTransport(respond)), **kwargs)
            state.clients.append(client)
            return client

        return create

    for name in ("OpenAI", "AsyncOpenAI", "AzureOpenAI", "AsyncAzureOpenAI"):
        monkeypatch.setattr(openai, name, factory(getattr(openai, name), name.startswith("Async")))
    return state


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("model", ["text-embedding-3-small", "openai/text-embedding-3-small"])
async def test_hosted_embedding_cache_and_batches(cache, sdk, asynchronous, model):
    embedding = Embedder(model, api_key="test-key", batch_size=2)
    call = embedding.acall if asynchronous else embedding

    async def invoke(**kwargs):
        result = call(["hello", "world!", "!"], **kwargs)
        return await result if asynchronous else result

    expected = [[5, 1, 2], [6, 1, 2], [1, 1, 2]]
    result = await invoke()
    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float32
    assert [json.loads(request.content)["input"] for request in sdk.requests] == [["hello", "world!"], ["!"]]
    assert all(json.loads(request.content)["model"] == "text-embedding-3-small" for request in sdk.requests)
    assert all(json.loads(request.content)["encoding_format"] == "float" for request in sdk.requests)
    assert all(str(request.url) == "https://api.openai.com/v1/embeddings" for request in sdk.requests)
    assert all(request.headers["authorization"] == "Bearer test-key" for request in sdk.requests)
    assert all(client.is_closed() for client in sdk.clients)

    np.testing.assert_allclose(await invoke(), expected)
    assert len(sdk.requests) == 2
    dspy.cache.reset_memory_cache()
    np.testing.assert_allclose(await invoke(), expected)
    assert len(sdk.requests) == 2
    np.testing.assert_allclose(await invoke(caching=False), expected)
    assert len(sdk.requests) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_call_overrides_and_request_options(cache, sdk, asynchronous):
    embedding = Embedder(
        "openai/org/model",
        caching=False,
        batch_size=1,
        api_base="https://first.example/v1",
        api_key="default-key",
        dimensions=12,
        timeout=30,
        max_retries=0,
    )
    kwargs = {
        "api_base": "https://second.example/v1",
        "api_key": "override-key",
        "dimensions": 3,
        "encoding_format": "float",
        "user": "test-user",
        "extra_body": {"custom": True},
        "extra_headers": {"X-Test": "yes"},
        "extra_query": {"custom-query": "yes"},
        "timeout": 12,
        "organization": "test-org",
        "project": "test-project",
        "caching": True,
        "batch_size": 3,
    }
    for _ in range(2):
        result = (
            await embedding.acall(["hello", "!"], **kwargs) if asynchronous else embedding(["hello", "!"], **kwargs)
        )
        np.testing.assert_allclose(result, [[5, 1, 2], [1, 1, 2]])
    assert len(sdk.requests) == 1
    request = sdk.requests[0]
    assert str(request.url) == "https://second.example/v1/embeddings?custom-query=yes"
    assert request.headers["authorization"] == "Bearer override-key"
    assert request.headers["x-test"] == "yes"
    assert request.headers["openai-organization"] == "test-org"
    assert request.headers["openai-project"] == "test-project"
    assert json.loads(request.content) == {
        "model": "org/model",
        "input": ["hello", "!"],
        "dimensions": 3,
        "encoding_format": "float",
        "user": "test-user",
        "custom": True,
    }
    assert sdk.options[0][1]["timeout"] == 12
    assert sdk.options[0][1]["max_retries"] == 0
    assert embedding.default_kwargs["api_key"] == "default-key"
    assert embedding.default_kwargs["dimensions"] == 12


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "model,default_option,endpoint_option",
    [
        ("openai/model", None, "api_base"),
        ("openai/model", None, "base_url"),
        ("openai/model", "api_base", "base_url"),
        ("openai/model", "base_url", "api_base"),
        ("azure/model", "api_base", "azure_endpoint"),
        ("azure/model", "api_base", "base_url"),
        ("azure/model", "azure_endpoint", "api_base"),
        ("azure/model", "azure_endpoint", "base_url"),
        ("azure/model", "base_url", "api_base"),
        ("azure/model", "base_url", "azure_endpoint"),
    ],
)
async def test_endpoints_are_part_of_cache_identity(cache, sdk, asynchronous, model, default_option, endpoint_option):
    def endpoint(host, option):
        suffix = ("/openai" if option == "base_url" else "") if model.startswith("azure/") else "/v1"
        return f"https://{host}.example{suffix}"

    defaults = {default_option: endpoint("default", default_option)} if default_option else {}
    if model.startswith("azure/"):
        defaults["api_version"] = "2024-02-01"
    embedding = Embedder(model, api_key="test-key", **defaults)
    for host in ("first", "second", "first"):
        kwargs = {endpoint_option: endpoint(host, endpoint_option)}
        if asynchronous:
            result = await embedding.acall("hello", **kwargs)
        else:
            result = embedding("hello", **kwargs)
        np.testing.assert_allclose(result, [5, 1, 2])
    assert [request.url.host for request in sdk.requests] == ["first.example", "second.example"]
    if default_option:
        if asynchronous:
            await embedding.acall("hello")
        else:
            embedding("hello")
        assert sdk.requests[-1].url.host == "default.example"


@pytest.mark.parametrize("endpoint_env", ["OPENAI_BASE_URL", "OPENAI_API_BASE"])
def test_environment_endpoint_is_part_of_cache_identity(cache, sdk, monkeypatch, endpoint_env):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    embedding = Embedder("text-embedding-3-small")
    for endpoint in ("https://first.example/v1", "https://second.example/v1"):
        monkeypatch.setenv(endpoint_env, endpoint)
        result = embedding("hello")
        assert result.shape == (3,)
    assert [request.url.host for request in sdk.requests] == ["first.example", "second.example"]
    assert all(request.headers["authorization"] == "Bearer env-key" for request in sdk.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("endpoint_option", ["api_base", "azure_endpoint", "base_url"])
async def test_azure_embeddings(cache, sdk, asynchronous, endpoint_option):
    endpoint = "https://azure.example" + ("/openai" if endpoint_option == "base_url" else "")
    embedding = Embedder(
        "azure/my-deployment",
        api_key="azure-key",
        api_version="2024-02-01",
        **{endpoint_option: endpoint},
    )
    result = await embedding.acall("hello") if asynchronous else embedding("hello")
    np.testing.assert_allclose(result, [5, 1, 2])
    request = sdk.requests[0]
    assert (
        str(request.url) == "https://azure.example/openai/deployments/my-deployment/embeddings?api-version=2024-02-01"
    )
    assert request.headers["api-key"] == "azure-key"
    assert json.loads(request.content)["model"] == "my-deployment"
    assert sdk.options[0][0] == ("AsyncAzureOpenAI" if asynchronous else "AzureOpenAI")
    assert sdk.clients[0].is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("credential", ["azure_ad_token", "azure_ad_token_provider", "environment"])
@pytest.mark.parametrize("key_env", ["AZURE_OPENAI_API_KEY", "AZURE_API_KEY"])
async def test_azure_token_does_not_add_environment_key(sdk, monkeypatch, asynchronous, credential, key_env):
    monkeypatch.setenv(key_env, "ambient-key")
    auth = {credential: "test-token" if credential == "azure_ad_token" else lambda: "test-token"}
    if credential == "environment":
        monkeypatch.setenv("AZURE_OPENAI_AD_TOKEN", "test-token")
        auth = {}
    embedding = Embedder("azure/my-deployment", api_base="https://azure.example", api_version="2024-02-01",
                         caching=False, **auth)
    result = await embedding.acall("hello") if asynchronous else embedding("hello")
    np.testing.assert_allclose(result, [5, 1, 2])
    assert sdk.requests[0].headers["authorization"] == "Bearer test-token"
    # Newer SDKs reject two explicit credential kinds, even though older ones accept them.
    assert sdk.options[0][1].get("api_key") is None
    assert sdk.clients[0].is_closed()


@pytest.mark.parametrize("legacy_env", [False, True])
def test_azure_environment_and_cache_identity(cache, sdk, monkeypatch, legacy_env):
    endpoint_env = "AZURE_API_BASE" if legacy_env else "AZURE_OPENAI_ENDPOINT"
    monkeypatch.setenv("AZURE_API_KEY" if legacy_env else "AZURE_OPENAI_API_KEY", "azure-env-key")
    version_env = "AZURE_API_VERSION" if legacy_env else "OPENAI_API_VERSION"
    embedding = Embedder("azure/my-deployment")
    for endpoint, version in (
        ("https://first.example", "2024-02-01"),
        ("https://second.example", "2024-02-01"),
        ("https://second.example", "2024-06-01"),
        ("https://second.example", "2024-06-01"),
    ):
        monkeypatch.setenv(endpoint_env, endpoint)
        monkeypatch.setenv(version_env, version)
        embedding("hello")
    assert len(sdk.requests) == 3
    assert [request.url.host for request in sdk.requests] == ["first.example", "second.example", "second.example"]
    assert sdk.requests[-1].url.params["api-version"] == "2024-06-01"
    assert all(request.headers["api-key"] == "azure-env-key" for request in sdk.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_explicit_base64_returns_numpy(sdk, asynchronous):
    embedding = Embedder("openai/model", api_key="test-key", encoding_format="base64", caching=False)
    result = await embedding.acall("hello") if asynchronous else embedding("hello")
    np.testing.assert_allclose(result, [5, 1, 2])
    assert result.dtype == np.float32


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_owned_clients_close_on_error(sdk, asynchronous):
    sdk.status_code = 400
    embedding = Embedder("openai/model", api_key="test-key", max_retries=0, caching=False)
    with pytest.raises(openai.BadRequestError):
        if asynchronous:
            await embedding.acall("hello")
        else:
            embedding("hello")
    assert len(sdk.requests) == 1
    assert sdk.clients[0].is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "model", ["cohere/embed-english-v3.0", "bedrock/amazon.titan-embed-text-v1", "openai/", "azure/"]
)
async def test_unsupported_prefix_rejected_before_client_creation(sdk, asynchronous, model):
    embedding = Embedder(model)
    with pytest.raises(ValueError, match="Hosted embeddings support"):
        if asynchronous:
            await embedding.acall("hello")
        else:
            embedding("hello")
    assert not sdk.clients


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_callable_embedding(cache, asynchronous):
    calls = []

    def embedding_fn(texts, scale=1):
        calls.append(texts)
        return [[len(text) * scale, 1, 2] for text in texts]

    embedding = Embedder(embedding_fn, batch_size=2, scale=2)
    for _ in range(2):
        result = (
            await embedding.acall(["hello", "world", "!"], scale=3)
            if asynchronous
            else embedding(["hello", "world", "!"], scale=3)
        )
        np.testing.assert_allclose(result, [[15, 1, 2], [15, 1, 2], [3, 1, 2]])
        assert result.dtype == np.float32
    assert calls == [["hello", "world"], ["!"]]


def test_callable_numpy_embedding_persists_to_disk(cache, tmp_path):
    dspy.cache.disk_cache.close()
    dspy.configure_cache(disk_cache_dir=tmp_path / ".dspy_cache_safe", restrict_pickle=True)
    inputs = ["hello", "world"]
    expected_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    embedding_fn = MagicMock(return_value=expected_embeddings)
    embedding = Embedder(embedding_fn)

    np.testing.assert_allclose(embedding(inputs), expected_embeddings)
    np.testing.assert_allclose(embedding(inputs), expected_embeddings)
    dspy.cache.reset_memory_cache()
    np.testing.assert_allclose(embedding(inputs), expected_embeddings)
    assert embedding_fn.call_count == 1


def test_invalid_model_type():
    with pytest.raises(ValueError, match="must be a string or a callable"):
        Embedder(123)(["test"])


def test_invalid_input():
    with pytest.raises(ValueError, match="All inputs must be strings"):
        Embedder(lambda texts: [])([123])
