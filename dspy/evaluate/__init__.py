from dspy.evaluate.metrics import EM, answer_exact_match, answer_passage_match, normalize_text

__all__ = [
    "EM",
    "normalize_text",
    "answer_exact_match",
    "answer_passage_match",
    "Evaluate",
    "SemanticF1",
    "CompleteAndGrounded",
    "EvaluationResult",
]

# Lazy imports — avoid pulling in heavy modules (IPython, ChainOfThought, etc.)
# at import time when only lightweight metric functions are needed.
_LAZY_IMPORTS = {
    "Evaluate": "dspy.evaluate.evaluate",
    "EvaluationResult": "dspy.evaluate.evaluate",
    "SemanticF1": "dspy.evaluate.auto_evaluation",
    "CompleteAndGrounded": "dspy.evaluate.auto_evaluation",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
