import litellm
import numpy as np
from typing import Callable, List, Union, Optional

class Embedder:
    """DSPy embedding class.

    The class for computing embeddings for text inputs. This class provides a unified interface for both:

    1. Hosted embedding models (e.g. OpenAI's text-embedding-3-small) via litellm integration
    2. Custom embedding functions that you provide

    For hosted models, simply pass the model name as a string (e.g. "openai/text-embedding-3-small"). The class will use
    litellm to handle the API calls and caching.

    For custom embedding models, pass a callable function to `embedding_function` that:
    - Takes a list of strings as input.
    - Returns embeddings as either:
        - A 2D numpy array of float32 values
        - A 2D list of float32 values
    - Each row should represent one embedding vector

    Args:
        embedding_model: The embedding model to use, either a string (for hosted models supported by litellm) or 
            a callable function that returns custom embeddings.
        embedding_function: An optional custom embedding function. If not provided, defaults to litellm
            for hosted models when `embedding_model` is a string.

    Examples:
        Example 1: Using a hosted model.

        ```python
        import dspy

        embedder = dspy.Embedding(embedding_model="openai/text-embedding-3-small")
        embeddings = embedder(["hello", "world"])

        assert embeddings.shape == (2, 1536)
        ```

        Example 2: Using a custom function.

        ```python
        import dspy
        import numpy as np

        def my_embedder(texts):
            return np.random.rand(len(texts), 10)

        embedder = dspy.Embedding(embedding_function=my_embedder)
        embeddings = embedder(["hello", "world"])

        assert embeddings.shape == (2, 10)
        ```
    """

    def __init__(self, embedding_model: Union[str, Callable[[List[str]], List[List[float]]]] = 'text-embedding-ada-002', embedding_function: Optional[Callable[[List[str]], List[List[float]]]] = None):
        self.embedding_model = embedding_model
        self.embedding_function = embedding_function or self.default_embedding_function

    def default_embedding_function(self, texts: List[str], caching: bool = True, **kwargs) -> List[List[float]]:
        embeddings_response = litellm.embedding(model=self.embedding_model, input=texts, caching=caching, **kwargs)
        return [data['embedding'] for data in embeddings_response.data]

    def __call__(self, inputs: Union[str, List[str]], caching: bool = True, **kwargs) -> np.ndarray:
        """Compute embeddings for the given inputs.

        Args:
            inputs: The inputs to compute embeddings for, can be a single string or a list of strings.
            caching: Whether to cache the embedding response, only valid when using a hosted embedding model.
            kwargs: Additional keyword arguments to pass to the embedding model.

        Returns:
            A 2-D numpy array of embeddings, one embedding per row.
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        if callable(self.embedding_function):
            embeddings = self.embedding_function(inputs, **kwargs)
        elif isinstance(self.embedding_model, str):
            embeddings = self.default_embedding_function(inputs, caching=caching, **kwargs)
        else:
            raise ValueError(
                f"`embedding_model` must be a string or `embedding_function` must be a callable, but got types: `embedding_model`={type(self.embedding_model)}, `embedding_function`={type(self.embedding_function)}."
            )
        return np.array(embeddings, dtype=np.float32)
