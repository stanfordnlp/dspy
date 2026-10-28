from __future__ import annotations

import base64
import os
from typing import Any, Callable

import openai

from dspy.clients.cache import request_cache
from dspy.utils.lazy_import import require

np = require("numpy")


class Embedder:
    """DSPy embedding class.

    The class for computing embeddings for text inputs. This class provides a unified interface for both:

    1. Hosted embedding models via the OpenAI SDK (including OpenAI-compatible endpoints and Azure OpenAI)
    2. Custom embedding functions that you provide

    For OpenAI models, pass a bare model name or an "openai/"-prefixed name, such as
    "openai/text-embedding-3-small". Use api_base (or base_url) for OpenAI-compatible endpoints.
    For Azure OpenAI, use "azure/<deployment>" with api_base (or azure_endpoint) and api_version.
    Other provider prefixes are not supported; use a custom callable for other APIs.
    Hosted requests use DSPy's cache and close their SDK clients after each uncached batch.

    For custom embedding models, pass a callable function that:
    - Takes a list of strings as input.
    - Returns embeddings as either:
        - A 2D numpy array of float32 values
        - A 2D list of float32 values
    - Each row should represent one embedding vector

    Args:
        model: A bare OpenAI model name, an "openai/<model>" or "azure/<deployment>" string,
            or a callable that represents a custom embedding model.
        batch_size (int, optional): The default batch size for processing inputs in batches. Defaults to 200.
        caching (bool, optional): Whether to cache the embedding response when using a hosted model. Defaults to True.
        **kwargs: Additional default keyword arguments to pass to the embedding model.

    Examples:
        Example 1: Using a hosted model.

        ```python
        import dspy

        embedder = dspy.Embedder("openai/text-embedding-3-small", batch_size=100)
        embeddings = embedder(["hello", "world"])

        assert embeddings.shape == (2, 1536)
        ```

        Example 2: Using any local embedding model, e.g. from https://huggingface.co/models?library=sentence-transformers.

        ```python
        # pip install sentence_transformers
        import dspy
        from sentence_transformers import SentenceTransformer

        # Load an extremely efficient local model for retrieval
        model = SentenceTransformer("sentence-transformers/static-retrieval-mrl-en-v1", device="cpu")

        embedder = dspy.Embedder(model.encode)
        embeddings = embedder(["hello", "world"], batch_size=1)

        assert embeddings.shape == (2, 1024)
        ```

        Example 3: Using a custom function.

        ```python
        import dspy
        import numpy as np

        def my_embedder(texts):
            return np.random.rand(len(texts), 10)

        embedder = dspy.Embedder(my_embedder)
        embeddings = embedder(["hello", "world"], batch_size=1)

        assert embeddings.shape == (2, 10)
        ```
    """

    def __init__(self, model: str | Callable, batch_size: int = 200, caching: bool = True, **kwargs: dict[str, Any]):
        self.model = model
        self.batch_size = batch_size
        self.caching = caching
        self.default_kwargs = kwargs

    def _preprocess(self, inputs, batch_size=None, caching=None, **kwargs):
        if isinstance(inputs, str):
            is_single_input = True
            inputs = [inputs]
        else:
            is_single_input = False

        if not all(isinstance(inp, str) for inp in inputs):
            raise ValueError("All inputs must be strings.")

        batch_size = batch_size or self.batch_size
        caching = caching if caching is not None else self.caching
        merged_kwargs = {**self.default_kwargs, **kwargs}
        if isinstance(self.model, str):
            if "/" in self.model:
                provider, model_name = self.model.split("/", 1)
                if provider not in ("openai", "azure") or not model_name:
                    raise ValueError(
                        "Hosted embeddings support bare OpenAI model names, 'openai/<model>', and 'azure/<deployment>' only. "
                        "For other providers, use an OpenAI-compatible endpoint with 'openai/<model>' or a callable. "
                        "Provider-specific options may require a callable; for Cohere retrieval, see "
                        "https://dspy.ai/api/models/Embedder/#retaining-a-cohere-retrieval-index."
                    )

            azure = self.model.startswith("azure/")
            endpoint_keys = ("api_base", "base_url", "azure_endpoint") if azure else ("api_base", "base_url")
            # Call-time endpoints replace default-only aliases. Conflicting
            # aliases within the call must still fail instead of choosing a host.
            if any(kwargs.get(key) is not None for key in endpoint_keys):
                for key in set(endpoint_keys) - kwargs.keys():
                    merged_kwargs.pop(key, None)
            if merged_kwargs.get("api_base") is not None and any(
                merged_kwargs.get(key) is not None for key in endpoint_keys[1:]
            ):
                if azure:
                    raise ValueError("Specify only one of api_base, azure_endpoint, or base_url for Azure embeddings.")
                raise ValueError("Specify only one of api_base or base_url for OpenAI embeddings.")
            endpoint = merged_kwargs.pop("api_base", None)
            if endpoint is not None:
                merged_kwargs["azure_endpoint" if azure else "base_url"] = endpoint
            if azure:
                environment = {
                    "api_version": os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_API_VERSION"),
                }
                if (
                    merged_kwargs.get("azure_ad_token") is None
                    and merged_kwargs.get("azure_ad_token_provider") is None
                    and os.getenv("AZURE_OPENAI_AD_TOKEN") is None
                ):
                    environment["api_key"] = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
                if not merged_kwargs.get("base_url"):
                    environment["azure_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_API_BASE")
            else:
                environment = {
                    "base_url": os.getenv("OPENAI_BASE_URL")
                    or os.getenv("OPENAI_API_BASE")
                    or "https://api.openai.com/v1",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                }
            for key, value in environment.items():
                merged_kwargs[key] = merged_kwargs.get(key) or value
            # Do not require compatible endpoints to implement the SDK's implicit base64 optimization.
            merged_kwargs.setdefault("encoding_format", "float")

        input_batches = []
        for i in range(0, len(inputs), batch_size):
            input_batches.append(inputs[i : i + batch_size])

        return input_batches, caching, merged_kwargs, is_single_input

    def _postprocess(self, embeddings_list, is_single_input):
        embeddings = np.array(embeddings_list, dtype=np.float32)
        if is_single_input:
            return embeddings[0]
        return embeddings

    def __call__(
        self,
        inputs: str | list[str],
        batch_size: int | None = None,
        caching: bool | None = None,
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Compute embeddings for the given inputs.

        Args:
            inputs: The inputs to compute embeddings for, can be a single string or a list of strings.
            batch_size (int, optional): The batch size for processing inputs. If None, defaults to the batch_size set
                during initialization.
            caching (bool, optional): Whether to cache the embedding response when using a hosted model. If None,
                defaults to the caching setting from initialization.
            kwargs: Additional keyword arguments to pass to the embedding model. These will override the default
                kwargs provided during initialization.

        Returns:
            numpy.ndarray: If the input is a single string, returns a 1D numpy array representing the embedding.
            If the input is a list of strings, returns a 2D numpy array of embeddings, one embedding per row.
        """
        input_batches, caching, kwargs, is_single_input = self._preprocess(inputs, batch_size, caching, **kwargs)

        compute_embeddings = _cached_compute_embeddings if caching else _cached_compute_embeddings.__wrapped__

        embeddings_list = []

        for batch in input_batches:
            embeddings_list.extend(compute_embeddings(self.model, batch, caching=caching, **kwargs))
        return self._postprocess(embeddings_list, is_single_input)

    async def acall(self, inputs, batch_size=None, caching=None, **kwargs):
        input_batches, caching, kwargs, is_single_input = self._preprocess(inputs, batch_size, caching, **kwargs)

        embeddings_list = []
        acompute_embeddings = _cached_acompute_embeddings if caching else _cached_acompute_embeddings.__wrapped__

        for batch in input_batches:
            embeddings_list.extend(await acompute_embeddings(self.model, batch, caching=caching, **kwargs))
        return self._postprocess(embeddings_list, is_single_input)


def _embedding_client(model, kwargs, asynchronous=False):
    client_keys = {
        "api_key",
        "base_url",
        "organization",
        "project",
        "timeout",
        "max_retries",
        "default_headers",
        "default_query",
    }
    if model.startswith("azure/"):
        client_type = openai.AsyncAzureOpenAI if asynchronous else openai.AzureOpenAI
        client_keys.update({"azure_endpoint", "api_version", "azure_ad_token", "azure_ad_token_provider"})
    else:
        client_type = openai.AsyncOpenAI if asynchronous else openai.OpenAI
    client_kwargs = {key: kwargs.pop(key) for key in client_keys if key in kwargs}
    return client_type(**client_kwargs)


def _response_embeddings(response):
    # The SDK decodes its default base64 response, but an explicit encoding_format="base64" is left encoded.
    return [
        np.frombuffer(base64.b64decode(data.embedding), dtype="<f4").tolist()
        if isinstance(data.embedding, str)
        else data.embedding
        for data in sorted(response.data, key=lambda data: data.index)
    ]


# Preserve these names and the caching argument: both are part of persisted
# cache keys. Callers use __wrapped__ to bypass caching, not SDK cleanup.
@request_cache(ignored_args_for_cache_key=["api_key", "azure_ad_token", "azure_ad_token_provider"])
def _cached_compute_embeddings(model, batch_inputs, caching=True, **kwargs):
    if isinstance(model, str):
        with _embedding_client(model, kwargs) as client:
            response = client.embeddings.create(model=model.split("/", 1)[-1], input=batch_inputs, **kwargs)
        return _response_embeddings(response)
    elif callable(model):
        return model(batch_inputs, **kwargs)
    else:
        raise ValueError(f"`model` in `dspy.Embedder` must be a string or a callable, but got {type(model)}.")


@request_cache(ignored_args_for_cache_key=["api_key", "azure_ad_token", "azure_ad_token_provider"])
async def _cached_acompute_embeddings(model, batch_inputs, caching=True, **kwargs):
    if not isinstance(model, str):
        return _cached_compute_embeddings.__wrapped__(model, batch_inputs, caching=caching, **kwargs)
    async with _embedding_client(model, kwargs, asynchronous=True) as client:
        response = await client.embeddings.create(model=model.split("/", 1)[-1], input=batch_inputs, **kwargs)
    return _response_embeddings(response)
