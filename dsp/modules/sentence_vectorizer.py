import abc
from typing import List

import numpy as np

from dsp.utils import determine_devices


class BaseSentenceVectorizer(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, inp_examples: List["Example"]) -> np.ndarray:
        pass


class SentenceTransformersVectorizer(BaseSentenceVectorizer):
    def __init__(
        self,
        model_name_or_path: str = 'all-MiniLM-L6-v2',
        vectorize_bs: int = 256,
        max_gpu_devices: int = 1,
        normalize_embeddings: bool = False
    ):
        # this isn't a good practice, but with top-level import the whole DSP
        # module import will be slow (>5 sec), because SentenceTransformer is doing
        # it's directory/file-related magic under the hood :(
        from sentence_transformers import SentenceTransformer
        self.num_devices, self.is_gpu = determine_devices(max_gpu_devices)
        self.proxy_device = 'cuda' if self.is_gpu else 'cpu'

        self.model = SentenceTransformer(model_name_or_path, device=self.proxy_device)

        self.model_name_or_path = model_name_or_path
        self.vectorize_bs = vectorize_bs
        self.normalize_embeddings = normalize_embeddings

    def __call__(self, inp_examples: List["Example"]) -> np.ndarray:
        text_to_vectorize = [example.text_to_vectorize for example in inp_examples]
        if self.is_gpu and self.num_devices > 1:
            target_devices = list(range(self.num_devices))
            pool = self.model.start_multi_process_pool(target_devices=target_devices)
            # Compute the embeddings using the multi-process pool
            emb = self.model.encode_multi_process(
                sentences=text_to_vectorize,
                pool=pool,
                batch_size=self.vectorize_bs
            )
            self.model.stop_multi_process_pool(pool)
            # for some reason, multi-GPU setup doesn't accept normalize_embeddings parameter
            if self.normalize_embeddings:
                emb = emb / np.linalg.norm(emb)

            return emb
        else:
            emb = self.model.encode(
                sentences=text_to_vectorize,
                batch_size=self.vectorize_bs,
                normalize_embeddings=self.normalize_embeddings
            )
            return emb


class NaiveGetFieldVectorizer(BaseSentenceVectorizer):
    def __init__(self, field_with_embeding: str = 'vectorized'):
        self.field_with_embeding = field_with_embeding

    def __call__(self, inp_examples: List["Example"]) -> np.ndarray:
        embeddings = [
            getattr(cur_example, self.field_with_embeding).reshape(1, -1)
            for cur_example in inp_examples
        ]
        embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
        return embeddings
