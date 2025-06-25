from typing import Any, Callable

import litellm
import numpy as np

from dspy.clients.cache import request_cache


class Embedder:
    """DSPy embedding class.

    The class for computing embeddings for text inputs. This class provides a unified interface for both:

    1. Hosted embedding models (e.g. OpenAI's text-embedding-3-small) via litellm integration
    2. Custom embedding functions that you provide

    For hosted models, simply pass the model name as a string (e.g., "openai/text-embedding-3-small"). The class will use
    litellm to handle the API calls and caching.

    For custom embedding models, pass a callable function that:
    - Takes a list of strings as input.
    - Returns embeddings as either:
        - A 2D numpy array of float32 values
        - A 2D list of float32 values
    - Each row should represent one embedding vector

    Args:
        model: The embedding model to use. This can be either a string (representing the name of the hosted embedding
            model, must be an embedding model supported by litellm) or a callable that represents a custom embedding
            model.
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
        caching = caching or self.caching
        merged_kwargs = self.default_kwargs.copy()
        merged_kwargs.update(kwargs)

        input_batches = []
        for i in range(0, len(inputs), batch_size):
            input_batches.append(inputs[i : i + batch_size])

        return input_batches, caching, merged_kwargs, is_single_input

    def _postprocess(self, embeddings_list, is_single_input):
        embeddings = np.array(embeddings_list, dtype=np.float32)
        if is_single_input:
            return embeddings[0]
        else:
            return np.array(embeddings, dtype=np.float32)

    def __call__(self, inputs: str | list[str], batch_size: int | None = None, caching: bool | None = None, **kwargs: dict[str, Any]) -> np.ndarray:
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

        compute_embeddings = _cached_compute_embeddings if caching else _compute_embeddings

        embeddings_list = []

        for batch in input_batches:
            embeddings_list.extend(compute_embeddings(self.model, batch, caching=caching, **kwargs))
        return self._postprocess(embeddings_list, is_single_input)

    async def acall(self, inputs, batch_size=None, caching=None, **kwargs):
        input_batches, caching, kwargs, is_single_input = self._preprocess(inputs, batch_size, caching, **kwargs)

        embeddings_list = []
        acompute_embeddings = _cached_acompute_embeddings if caching else _acompute_embeddings

        for batch in input_batches:
            embeddings_list.extend(await acompute_embeddings(self.model, batch, caching=caching, **kwargs))
        return self._postprocess(embeddings_list, is_single_input)


def _compute_embeddings(model, batch_inputs, caching=False, **kwargs):
    if isinstance(model, str):
        caching = caching and litellm.cache is not None
        embedding_response = litellm.embedding(model=model, input=batch_inputs, caching=caching, **kwargs)
        return [data["embedding"] for data in embedding_response.data]
    elif callable(model):
        return model(batch_inputs, **kwargs)
    else:
        raise ValueError(f"`model` in `dspy.Embedder` must be a string or a callable, but got {type(model)}.")


@request_cache(ignored_args_for_cache_key=["api_key", "api_base", "base_url"])
def _cached_compute_embeddings(model, batch_inputs, caching=True, **kwargs):
    return _compute_embeddings(model, batch_inputs, caching=caching, **kwargs)


async def _acompute_embeddings(model, batch_inputs, caching=False, **kwargs):
    if isinstance(model, str):
        caching = caching and litellm.cache is not None
        embedding_response = await litellm.aembedding(model=model, input=batch_inputs, caching=caching, **kwargs)
        return [data["embedding"] for data in embedding_response.data]
    elif callable(model):
        return model(batch_inputs, **kwargs)
    else:
        raise ValueError(f"`model` in `dspy.Embedder` must be a string or a callable, but got {type(model)}.")


@request_cache(ignored_args_for_cache_key=["api_key", "api_base", "base_url"])
async def _cached_acompute_embeddings(model, batch_inputs, caching=True, **kwargs):
    return await _acompute_embeddings(model, batch_inputs, caching=caching, **kwargs)
