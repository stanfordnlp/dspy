from typing import List, Callable
import numpy as np
import dsp

class KNN(Module):
    def __init__(self, k: int, trainset: List[dsp.Example]):
        super().__init__()
        self.k = k
        self.trainset = trainset
        self.vectorizer = dsp.SentenceTransformersVectorizer()

    def __call__(self, **kwargs) -> List[dsp.Example]:
        with dsp.settings.context(vectorizer=self.vectorizer):
            trainset_casted_to_vectorize = [" ".join([example[key] for key in example._input_keys]) for example in self.trainset]
            all_vectors = self.vectorizer(trainset_casted_to_vectorize).astype(np.float32)
            inp_example_vector = self.vectorizer([" ".join([f"{key} {val}" for key, val in kwargs.items()])])
            scores = np.dot(all_vectors, inp_example_vector.T).squeeze()
            nearest_samples_idxs = scores.argsort()[-self.k:][::-1]
            train_sampled = [self.trainset[cur_idx] for cur_idx in nearest_samples_idxs]
            return train_sampled