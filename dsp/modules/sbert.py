class SentenceTransformersCrossEncoder:
    """Wrapper for sentence-transformers cross-encoder model.
    """
    def __init__(
        self, model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    ):
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
        except ImportError:
            raise ModuleNotFoundError(
                "You need to install sentence-transformers library to use SentenceTransformersCrossEncoder.",
            )
        self.model = CrossEncoder(model_name_or_path)

    def __call__(self, query: str, passage: list[str]) -> list[float]:
        return self.model.predict([[query, p] for p in passage]).tolist()
