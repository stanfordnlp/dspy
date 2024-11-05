import litellm
import numpy as np


class Embedding:
    """DSPy embedding class.

    The class for computing embeddings for text inputs. This class provides a unified interface for both:

    1. Hosted embedding models (e.g. OpenAI's text-embedding-3-small) via litellm integration
    2. Custom embedding functions that you provide

    For hosted models, simply pass the model name as a string (e.g. "openai/text-embedding-3-small"). The class will use
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

    Examples:
        Example 1: Using a hosted model.

        ```python
        import dspy

        embedder = dspy.Embedding("openai/text-embedding-3-small")
        embeddings = embedder(["hello", "world"])

        assert embeddings.shape == (2, 1536)
        ```

        Example 2: Using a custom function.

        ```python
        import dspy

        def my_embedder(texts):
            return np.random.rand(len(texts), 10)

        embedder = dspy.Embedding(my_embedder)
        embeddings = embedder(["hello", "world"])

        assert embeddings.shape == (2, 10)
        ```
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, inputs, caching=True, **kwargs):
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
        if isinstance(self.model, str):
            embedding_response = litellm.embedding(model=self.model, input=inputs, caching=caching, **kwargs)
            return np.array([data["embedding"] for data in embedding_response.data], dtype=np.float32)
        elif callable(self.model):
            return np.array(self.model(inputs, **kwargs), dtype=np.float32)
        else:
            raise ValueError(f"`model` in `dspy.Embedding` must be a string or a callable, but got {type(self.model)}.")
