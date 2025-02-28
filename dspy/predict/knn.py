from typing import Callable

import numpy as np


class KNN:

    def __init__(self, k: int, trainset: list,
                 vectorizer: Callable[[list[str]], np.ndarray]):
        """
        A k-nearest neighbors retriever that finds similar examples from a training set.

        Args:
            k: Number of nearest neighbors to retrieve
            trainset: List of training examples to search through
            vectorizer: A callable vectorizer for computing embeddings.

        Example:
            >>> trainset = [dsp.Example(input="hello", output="world"), ...]
            >>> knn = KNN(k=3, trainset=trainset, vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode))
            >>> similar_examples = knn(input="hello")
        """
        self.k = k
        self.trainset = trainset
        self.embedding = vectorizer
        trainset_casted_to_vectorize = [
            " | ".join([f"{key}: {value}" for key, value in example.items() if key in example._input_keys])
            for example in self.trainset
        ]
        self.trainset_vectors = self.embedding(trainset_casted_to_vectorize).astype(np.float32)

    def __call__(self, **kwargs) -> list:
        input_example_vector = self.embedding([" | ".join([f"{key}: {val}" for key, val in kwargs.items()])])
        scores = np.dot(self.trainset_vectors, input_example_vector.T).squeeze()
        nearest_samples_idxs = scores.argsort()[-self.k :][::-1]
        train_sampled = [self.trainset[cur_idx] for cur_idx in nearest_samples_idxs]
        return train_sampled
