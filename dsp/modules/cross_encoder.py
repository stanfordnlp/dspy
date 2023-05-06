from abc import ABC, abstractmethod

class BaseCrossEncoder(ABC):
    '''
    Base class for Cross Encoders. The main purpose is to compute similarity between
    two text inputs. `__call__` method takes `str` and `List[str]` as a single input, then
    '''

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, query: str, passage: list[str]):
        pass
    
    
class SentenceTransformersCrossEncoder(BaseCrossEncoder):
    def __init__(self, model_name_or_path: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2'):
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "You need to install sentence_transformers library to use pretrained embedders. "
                "Please check the official doc https://www.sbert.net/ "
                "or simply run `pip install sentence-transformers"
            )
        self.model = CrossEncoder(model_name_or_path)
        
    def __call__(self, query: str, passage: list[str]):
        return self.model.predict([(query, p) for p in passage])