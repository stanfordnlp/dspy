from functools import partial
from typing import Callable, List, Optional, Union
import litellm
import numpy as np

from .lm import request_cache


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

        Example 2: Using a custom function.

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

    def __init__(self, model: Union[str, Callable], batch_size=200, cache=True, **kwargs):
        if not isinstance(model, str) and not callable(model):
            raise ValueError(f"`model` in `dspy.Embedder` must be a string or a callable, but got {type(model)}.")

        self.model = model
        self.batch_size = batch_size
        self.cache = cache
        self.default_kwargs = kwargs

    def _embed(self, inputs: List[str], cache: bool, **kwargs):
        if callable(self.model):
            return self.model(inputs, **kwargs)

        response = litellm_embedding({"model": self.model, "input": inputs, **kwargs}, cache=cache).data
        return [data["embedding"] for data in response]

    def __call__(
        self,
        inputs: Union[str, List[str]],
        batch_size: Optional[int] = None,
        cache: Optional[bool] = None,
        **kwargs,
    ) -> np.ndarray:
        """Compute embeddings for the given inputs.

        Args:
            inputs: The inputs to compute embeddings for, can be a single string or a list of strings.
            batch_size (int, optional): The batch size for processing inputs. If None, defaults to the batch_size set during initialization.
            caching (bool, optional): Whether to cache the embedding response when using a hosted model. If None, defaults to the caching setting from initialization.
            **kwargs: Additional keyword arguments to pass to the embedding model. These will override the default kwargs provided during initialization.

        Returns:
            numpy.ndarray: If the input is a single string, returns a 1D numpy array representing the embedding.
            If the input is a list of strings, returns a 2D numpy array of embeddings, one embedding per row.
        """

        multi_input = isinstance(inputs, list)
        if not multi_input:
            inputs = [inputs]

        assert all(isinstance(inp, str) for inp in inputs), "All inputs must be strings."

        batch_size = batch_size or self.batch_size
        cache = cache or self.cache
        kwargs = {**self.default_kwargs, **kwargs}

        embeddings = flatten([self._embed(c, cache, **kwargs) for c in chunk(inputs, batch_size)])
        embeddings = embeddings if multi_input else embeddings[0]
        return np.array(embeddings, dtype=np.float32)


def chunk(inputs_list, size):
    for i in range(0, len(inputs_list), size):
        yield inputs_list[i : i + size]


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def litellm_embedding(request, cache=True):
    if not cache:
        return litellm.embedding(**request, cache={"no-cache": True, "no-store": True})

    response = request_cache().get(request, None)
    if response:
        return response

    response = litellm.embedding(**request, cache={"no-cache": False, "no-store": False})
    request_cache()[request] = response

    return response
