import abc
from typing import List, Optional
import numpy as np
import openai

try:
    import boto3
    from botocore.exceptions import ClientError
    ERRORS = (ClientError,)

except ImportError:
    ERRORS = (Exception,)

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
        self.client = cohere.Client(api_key, client_name='dspy')

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

class FastEmbedVectorizer(BaseSentenceVectorizer):
    """Sentence vectorizer implementaion using FastEmbed - https://qdrant.github.io/fastembed."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 256,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        parallel: Optional[int] = None,
        **kwargs,
    ):
        """Initialize fastembed.TextEmbedding.

        Args:
            model_name (str): The name of the model to use. Defaults to `"BAAI/bge-small-en-v1.5"`.
            batch_size (int): Batch size for encoding. Higher values will use more memory, but be faster.\
                                        Defaults to 256.
            cache_dir (str, optional): The path to the model cache directory.\
                                       Can also be set using the `FASTEMBED_CACHE_PATH` env variable.
            threads (int, optional): The number of threads single onnxruntime session can use.
            parallel (int, optional): If `>1`, data-parallel encoding will be used, recommended for large datasets.\
                                      If `0`, use all available cores.\
                                      If `None`, don't use data-parallel processing, use default onnxruntime threading.\
                                      Defaults to None.
            **kwargs: Additional options to pass to fastembed.TextEmbedding
        Raises:
            ValueError: If the model_name is not in the format <org>/<model> e.g. BAAI/bge-small-en-v1.5.
        """
        try:
            from fastembed import TextEmbedding
        except ImportError as e:
            raise ValueError(
                "The 'fastembed' package is not installed. Please install it with `pip install fastembed`",
            ) from e
        self._batch_size = batch_size
        self._parallel = parallel
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir, threads=threads, **kwargs)

    def __call__(self, inp_examples: List["Example"]) -> np.ndarray:
        texts_to_vectorize = self._extract_text_from_examples(inp_examples)
        embeddings = self._model.embed(texts_to_vectorize, batch_size=self._batch_size, parallel=self._parallel)

        return np.array([embedding.tolist() for embedding in embeddings], dtype=np.float32)

class AmazonBedrockVectorizer(BaseSentenceVectorizer):
    '''
    This vectorizer uses Amazon Bedrock API to convert texts to embeddings.
    '''
    SUPPORTED_MODELS = [
        "amazon.titan-embed-text-v1", "amazon.titan-embed-text-v2:0",
        "cohere.embed-english-v3", "cohere.embed-multilingual-v3"
    ]

    def __init__(
        self,
        model_id: str = 'amazon.titan-embed-text-v2:0',
        embed_batch_size: int = 128,
        region_name: str = 'us-west-2',
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        self.model_id = model_id
        self.embed_batch_size = embed_batch_size

        # Initialize the Bedrock client
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

    def __call__(self, inp_examples: List["Example"]) -> np.ndarray:
        text_to_vectorize = self._extract_text_from_examples(inp_examples)
        embeddings_list = []

        n_batches = (len(text_to_vectorize) - 1) // self.embed_batch_size + 1
        for cur_batch_idx in range(n_batches):
            start_idx = cur_batch_idx * self.embed_batch_size
            end_idx = (cur_batch_idx + 1) * self.embed_batch_size
            cur_batch = text_to_vectorize[start_idx: end_idx]
            
            # Configure Bedrock API Body
            if self.model_id not in self.SUPPORTED_MODELS:
                raise Exception(f"Unsupported model: {self.model_id}")
            
            if self.model_id == "amazon.titan-embed-text-v1":
                if self.embed_batch_size == 1:
                    body = json.dumps({"inputText": cur_batch[0]})
                else:
                    raise Exception(f"Model {self.model_id} supports batch size of 1 exclusively.")
            elif self.model_id == "amazon.titan-embed-text-v2:0":
                if self.embed_batch_size == 1:
                    body = json.dumps({
                        "inputText": cur_batch[0],
                        "dimensions": 512
                    })
                else:
                    raise Exception(f"Model {self.model_id} supports batch size of 1 exclusively.")
            elif self.model_id.startswith("cohere.embed"):
                body = json.dumps({
                    "texts": cur_batch,
                    "input_type": "search_document"
                })
            else:
                raise Exception(f"Unexpected state: model_id '{self.model_id}' is not properly handled. Please check the implementation for supported models.")
                
            
            # Invoke Bedrock API
            response = self.bedrock_client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept='application/json',
                contentType='application/json'
            )

            response_body = json.loads(response['body'].read())
            if self.model_id.startswith("cohere.embed"):
                cur_batch_embeddings = response_body['embeddings']
            elif self.model_id.startswith("amazon.titan-embed-text"):
                cur_batch_embeddings = response_body['embedding']
            else:
                raise Exception(f"Unexpected response format for model {self.model_id}. Check the Amazon Bedrock documentation for the correct response structure.")
            embeddings_list.extend(cur_batch_embeddings)

        embeddings = np.array(embeddings_list, dtype=np.float32)
        return embeddings

    def _extract_text_from_examples(self, inp_examples: List) -> List[str]:
        if isinstance(inp_examples[0], str):
            return inp_examples 
        return [" ".join([example[key] for key in example._input_keys]) for example in inp_examples]
