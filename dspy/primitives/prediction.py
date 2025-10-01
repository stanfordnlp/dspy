from __future__ import annotations

from typing import Any, Callable

from dspy.primitives.example import Example
from dspy.metrics import Score
from dspy.metrics._subscores import _begin_collect, _end_collect, finalize_scores


ScoreFn = Callable[[Example, "Prediction", Any], Any]


class Prediction(Example):
    """A prediction object that contains the output of a DSPy module.
    
    Prediction inherits from Example.
    
    To allow feedback-augmented scores, Prediction supports comparison operations
    (<, >, <=, >=) for Predictions with a `score` field. The comparison operations
    compare the 'score' values as floats. For equality comparison, Predictions are equal
    if their underlying data stores are equal (inherited from Example).
    
    Arithmetic operations (+, /, etc.) are also supported for Predictions with a 'score'
    field, operating on the score value.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        del self._demos
        del self._input_keys

        self._completions = None
        self._lm_usage = None
        self._score_value: float | None = None
        self._score_fn: ScoreFn | None = None
        self._scores_obj: Score | None = None
        self._bound_example: Example | None = None
        self._resolved_ctx_key: Any = None

        if "score" in self._store:
            initial_score = self._store.pop("score")
            if initial_score is not None:
                self.score = initial_score

    def get_lm_usage(self):
        return self._lm_usage

    def set_lm_usage(self, value):
        self._lm_usage = value

    @classmethod
    def from_completions(cls, list_or_dict, signature=None):
        obj = cls()
        obj._completions = Completions(list_or_dict, signature=signature)
        obj._store = {k: v[0] for k, v in obj._completions.items()}

        return obj

    def __repr__(self):
        store_repr = ",\n    ".join(f"{k}={v!r}" for k, v in self._store.items())

        if self._completions is None or len(self._completions) == 1:
            return f"Prediction(\n    {store_repr}\n)"

        num_completions = len(self._completions)
        return f"Prediction(\n    {store_repr},\n    completions=Completions(...)\n) ({num_completions-1} completions omitted)"

    def __str__(self):
        return self.__repr__()

    def __float__(self):
        return float(self._require_numeric_score())

    def __add__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() + other
        elif isinstance(other, Prediction):
            return self._require_numeric_score() + other._require_numeric_score()
        raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __radd__(self, other):
        if isinstance(other, (float, int)):
            return other + self.__float__()
        elif isinstance(other, Prediction):
            return other._require_numeric_score() + self._require_numeric_score()
        raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return self._require_numeric_score() / other
        elif isinstance(other, Prediction):
            return self._require_numeric_score() / other._require_numeric_score()
        raise TypeError(f"Unsupported type for division: {type(other)}")

    def __rtruediv__(self, other):
        if isinstance(other, (float, int)):
            return other / self._require_numeric_score()
        elif isinstance(other, Prediction):
            return other._require_numeric_score() / self._require_numeric_score()
        raise TypeError(f"Unsupported type for division: {type(other)}")

    def __lt__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() < other
        elif isinstance(other, Prediction):
            return self._require_numeric_score() < other._require_numeric_score()
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    def __le__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() <= other
        elif isinstance(other, Prediction):
            return self._require_numeric_score() <= other._require_numeric_score()
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    def __gt__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() > other
        elif isinstance(other, Prediction):
            return self._require_numeric_score() > other._require_numeric_score()
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    def __ge__(self, other):
        if isinstance(other, (float, int)):
            return self.__float__() >= other
        elif isinstance(other, Prediction):
            return self._require_numeric_score() >= other._require_numeric_score()
        raise TypeError(f"Unsupported type for comparison: {type(other)}")

    @property
    def completions(self):
        return self._completions

    @property
    def score(self) -> float | ScoreFn | None:
        if self._score_value is not None:
            return self._score_value
        return self._score_fn

    @score.setter
    def score(self, value: float | ScoreFn | None) -> None:
        self._scores_obj = None
        self._resolved_ctx_key = None
        if callable(value):
            self._score_fn = value
            self._score_value = None
            self._store.pop("score", None)
        else:
            self._score_fn = None
            if value is None:
                self._score_value = None
                self._store.pop("score", None)
            else:
                self._score_value = float(value)
                self._store["score"] = self._score_value

    @property
    def scores(self) -> Score | None:
        if self._scores_obj is None and self._score_value is not None:
            self._scores_obj = Score(self._score_value)
        return self._scores_obj

    def bind_example(self, example: Example) -> None:
        self._bound_example = example

    def resolve_score(self, ctx: Any | None = None, *, force: bool = False) -> Score | None:
        if self._score_fn is None:
            return self.scores

        ctx_key = getattr(ctx, "cache_key", None)
        if not force and self._scores_obj is not None and self._resolved_ctx_key == ctx_key:
            return self._scores_obj

        if self._bound_example is None:
            raise RuntimeError("Cannot resolve score: no bound example. Call bind_example(ex).")

        token = _begin_collect()
        try:
            result = self._score_fn(self._bound_example, self, ctx)
        finally:
            collector = _end_collect(token)

        ctx_info = {}
        if ctx is not None:
            usage = getattr(ctx, "usage", None)
            latency_ms = getattr(ctx, "latency_ms", None)
            seed = getattr(ctx, "seed", None)
            if usage is not None:
                ctx_info["usage"] = usage
            if latency_ms is not None:
                ctx_info["latency_ms"] = latency_ms
            if seed is not None:
                ctx_info["seed"] = seed

        scores = finalize_scores(result, collector, ctx_info=ctx_info)
        self._store_scores(scores, ctx_key)
        return scores

    def _require_numeric_score(self) -> float:
        if self._score_value is None:
            raise TypeError("Prediction score is not resolved to a numeric value.")
        return float(self._score_value)

    def _store_scores(self, scores: Score, ctx_key: Any | None = None) -> None:
        self._scores_obj = scores
        self._score_value = scores.scalar
        self._store["score"] = self._score_value
        self._resolved_ctx_key = ctx_key


class Completions:
    def __init__(self, list_or_dict, signature=None):
        self.signature = signature

        if isinstance(list_or_dict, list):
            kwargs = {}
            for arg in list_or_dict:
                for k, v in arg.items():
                    kwargs.setdefault(k, []).append(v)
        else:
            kwargs = list_or_dict

        assert all(isinstance(v, list) for v in kwargs.values()), "All values must be lists"

        if kwargs:
            length = len(next(iter(kwargs.values())))
            assert all(len(v) == length for v in kwargs.values()), "All lists must have the same length"

        self._completions = kwargs

    def items(self):
        return self._completions.items()

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0 or key >= len(self):
                raise IndexError("Index out of range")

            return Prediction(**{k: v[key] for k, v in self._completions.items()})

        return self._completions[key]

    def __getattr__(self, name):
        if name == "_completions":
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        if name in self._completions:
            return self._completions[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __len__(self):
        # Return the length of the list for one of the keys
        # It assumes all lists have the same length
        return len(next(iter(self._completions.values())))

    def __contains__(self, key):
        return key in self._completions

    def __repr__(self):
        items_repr = ",\n    ".join(f"{k}={v!r}" for k, v in self._completions.items())
        return f"Completions(\n    {items_repr}\n)"

    def __str__(self):
        # return str(self._completions)
        return self.__repr__()
