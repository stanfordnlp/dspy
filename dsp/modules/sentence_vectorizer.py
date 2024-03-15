import abc
from typing import List, Optional

import numpy as np
import openai


class BaseSentenceVectorizer(abc.ABC):
    '''
    Base Class for Vectorizers. The main purpose is to vectorize text (doc/query)
    for ANN/KNN indexes. `__call__` method takes `List[Example]` as a single input, then extracts
    `field_to_vectorize` from every Example and convert them into embeddings.
    You can customize extraction logic in the `_extract_text_from_examples` method.
    '''
    # embeddings will be computed based on the string in this attribute of Example object
    field_to_vectorize = 'text_to_vectorize'

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, inp_examples: List["Example"]) -> np.ndarray:
        pass

    def _extract_text_from_examples(self, inp_examples: List) -> List[str]:
        if isinstance(inp_examples[0], str):
            return inp_examples 
        return [" ".join([example[key] for key in example._input_keys]) for example in inp_examples]


class SentenceTransformersVectorizer(BaseSentenceVectorizer):
    '''
    Vectorizer based on `SentenceTransformers` models. You can pick any model from this link:
    https://huggingface.co/models?library=sentence-transformers
    More details about models:
    https://www.sbert.net/docs/pretrained_models.html
    '''
    def __init__(
        self,
        model_name_or_path: str = 'all-MiniLM-L6-v2',
        vectorize_bs: int = 256,
        max_gpu_devices: int = 1,
        normalize_embeddings: bool = False,
    ):
        # this isn't a good practice, but with top-level import the whole DSP
        # module import will be slow (>5 sec), because SentenceTransformer is doing
        # it's directory/file-related magic under the hood :(
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "You need to install sentence_transformers library to use pretrained embedders. "
                "Please check the official doc https://www.sbert.net/ "
                "or simply run `pip install sentence-transformers",
            )
        from dsp.utils.ann_utils import determine_devices
        
        self.num_devices, self.is_gpu = determine_devices(max_gpu_devices)
        self.proxy_device = 'cuda' if self.is_gpu else 'cpu'

        self.model = SentenceTransformer(model_name_or_path, device=self.proxy_device)

        self.model_name_or_path = model_name_or_path
        self.vectorize_bs = vectorize_bs
        self.normalize_embeddings = normalize_embeddings

    def __call__(self, inp_examples: List) -> np.ndarray:
        text_to_vectorize = self._extract_text_from_examples(inp_examples)

        if self.is_gpu and self.num_devices > 1:
            target_devices = list(range(self.num_devices))
            pool = self.model.start_multi_process_pool(target_devices=target_devices)
            # Compute the embeddings using the multi-process pool
            emb = self.model.encode_multi_process(
                sentences=text_to_vectorize,
                pool=pool,
                batch_size=self.vectorize_bs,
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
                normalize_embeddings=self.normalize_embeddings,
            )
            return emb


class NaiveGetFieldVectorizer(BaseSentenceVectorizer):
    '''
    If embeddings were precomputed, then we could just extract them from the proper field 
    (set by `field_with_embedding`) from each `Example`.
    '''
    def __init__(self, field_with_embedding: str = 'vectorized'):
        self.field_with_embedding = field_with_embedding

    def __call__(self, inp_examples: List["Example"]) -> np.ndarray:
        embeddings = [
            getattr(cur_example, self.field_with_embedding).reshape(1, -1)
            for cur_example in inp_examples
        ]
        embeddings = np.concatenate(embeddings, axis=0).astype(np.float32)
        return embeddings


class CohereVectorizer(BaseSentenceVectorizer):
    '''
    This vectorizer uses the Cohere API to convert texts to embeddings.
    More about the available models: https://docs.cohere.com/reference/embed
    `api_key` should be passed as an argument and can be retrieved
    from https://dashboard.cohere.com/api-keys
    '''
    def __init__(
        self,
        api_key: str,
        model: str = 'embed-english-v3.0',
        embed_batch_size: int = 96,
        embedding_type: str = 'search_document',  # for details check Cohere embed docs
    ):
        self.model = model
        self.embed_batch_size = embed_batch_size
        self.embedding_type = embedding_type

        import cohere
        self.client = cohere.Client(api_key)

    def __call__(self, inp_examples: List["Example"]) -> np.ndarray:
        text_to_vectorize = self._extract_text_from_examples(inp_examples)

        embeddings_list = []

        n_batches = (len(text_to_vectorize) - 1) // self.embed_batch_size + 1
        for cur_batch_idx in range(n_batches):
            start_idx = cur_batch_idx * self.embed_batch_size
            end_idx = (cur_batch_idx + 1) * self.embed_batch_size
            cur_batch = text_to_vectorize[start_idx: end_idx]

            response = self.client.embed(
                texts=cur_batch,
                model=self.model,
                input_type=self.embedding_type,
            )

            embeddings_list.extend(response.embeddings)

        embeddings = np.array(embeddings_list, dtype=np.float32)
        return embeddings


try:
    OPENAI_LEGACY = int(openai.version.__version__[0]) == 0
except Exception:
    OPENAI_LEGACY = True


class OpenAIVectorizer(BaseSentenceVectorizer):
    '''
    This vectorizer uses OpenAI API to convert texts to embeddings. Changing `model` is not
    recommended. More about the model: https://openai.com/blog/new-and-improved-embedding-model/
    `api_key` should be passed as an argument or as env variable (`OPENAI_API_KEY`).
    '''
    def __init__(
        self,
        model: str = 'text-embedding-ada-002',
        embed_batch_size: int = 1024,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.embed_batch_size = embed_batch_size

        if OPENAI_LEGACY:
            self.Embedding = openai.Embedding
        else:
            self.Embedding = openai.embeddings

        if api_key:
            openai.api_key = api_key

    def __call__(self, inp_examples: List["Example"]) -> np.ndarray:
        text_to_vectorize = self._extract_text_from_examples(inp_examples)
        # maybe it's better to preallocate numpy matrix, but we don't know emb_dim
        embeddings_list = []

        n_batches = (len(text_to_vectorize) - 1) // self.embed_batch_size + 1
        for cur_batch_idx in range(n_batches):  # tqdm.tqdm?
            start_idx = cur_batch_idx * self.embed_batch_size
            end_idx = (cur_batch_idx + 1) * self.embed_batch_size
            cur_batch = text_to_vectorize[start_idx: end_idx]
            # OpenAI API call:
            response = self.Embedding.create(
                model=self.model,
                input=cur_batch,
            )

            cur_batch_embeddings = [cur_obj['embedding'] for cur_obj in response['data']]
            embeddings_list.extend(cur_batch_embeddings)

        embeddings = np.array(embeddings_list, dtype=np.float32)
        return embeddings
