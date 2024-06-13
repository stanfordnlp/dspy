from typing import List, Optional

import numpy as np

import dsp


class KNN:
    def __init__(self, k: int, trainset: List[dsp.Example], vectorizer: Optional[dsp.BaseSentenceVectorizer] = None):
        self.k = k
        self.trainset = trainset
        self.vectorizer = vectorizer or dsp.SentenceTransformersVectorizer()
        trainset_casted_to_vectorize = [
            " | ".join([f"{key}: {value}" for key, value in example.items() if key in example._input_keys])
            for example in self.trainset
        ]
        self.trainset_vectors = self.vectorizer(trainset_casted_to_vectorize).astype(np.float32)

    def __call__(self, **kwargs) -> List[dsp.Example]:
        with dsp.settings.context(vectorizer=self.vectorizer):
            input_example_vector = self.vectorizer([" | ".join([f"{key}: {val}" for key, val in kwargs.items()])])
            scores = np.dot(self.trainset_vectors, input_example_vector.T).squeeze()
            nearest_samples_idxs = scores.argsort()[-self.k :][::-1]
            train_sampled = [self.trainset[cur_idx] for cur_idx in nearest_samples_idxs]
            return train_sampled
