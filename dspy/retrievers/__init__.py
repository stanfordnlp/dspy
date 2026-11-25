from dspy.retrievers.embeddings import Embeddings, EmbeddingsWithScores
from dspy.retrievers.retrieve import Retrieve

__all__ = ["Embeddings", "EmbeddingsWithScores", "Retrieve"]

# Optional integrations — imported lazily to avoid requiring extra packages at import time
def __getattr__(name: str):
    if name == "DakeraRM":
        from dspy.retrievers.dakera_rm import DakeraRM
        return DakeraRM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
