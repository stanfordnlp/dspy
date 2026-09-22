"""Decision values shared by System One predictors and generative LMs."""

import json
from functools import lru_cache
from typing import Annotated, ClassVar, Literal, get_args, get_origin

from pydantic import ConfigDict, Field, create_model, model_serializer
from pydantic.json_schema import SkipJsonSchema
from pydantic_core import core_schema

from dspy.adapters.types.base_type import Type
from dspy.utils.annotation import experimental

Probability = Annotated[float, Field(ge=0, le=1)]


class _Decision(Type):
    model_config = ConfigDict(extra="forbid")

    confidence: Probability = Field(description="Confidence in the value, from 0 to 1.")

    @model_serializer(mode="wrap")
    def serialize_model(self, handler):
        # Unlike media types, decisions are ordinary structured JSON, including
        # provider evidence when present. Missing evidence is not shown as null.
        return {key: value for key, value in handler(self).items() if value is not None or key == "value"}


@experimental
class Noul(_Decision):
    """A Boolean value with confidence and optional provider true-probability.

    Predict generates value and self-reported confidence. Decide derives value
    from the provider probability and its per-field threshold (default 0.5).
    Decide confidence is ``abs(p - threshold) / max(threshold, 1 - threshold)``:
    a distance from the decision boundary, not a calibrated probability.
    """

    value: bool = Field(strict=True)
    probability: SkipJsonSchema[Probability | None] = None

    @classmethod
    def description(cls):
        return "A Boolean value and confidence (0 to 1)."

    def __bool__(self):
        return self.value


@experimental
class Score(_Decision):
    """A continuous score with a declared rubric and confidence.

    Declare ``Score["poor", "fair", "excellent"]`` in increasing order.
    Predict generates a continuous value from 0 to N-1 and confidence;
    Decide averages the level indices using provider probabilities.
    ``probabilities`` retains the raw distribution, keyed by rubric index.
    ``level`` is the zero-based ordinal selected by Decide's cuts on the mean
    level index. It does not change the continuous value or provider confidence.
    Predict generates neither probabilities nor level.

    ``Annotated[float, Score[...]]`` uses the same rubric but returns a native
    float instead of a rich result. Bare float has no rubric for Decide.
    """

    value: float = Field(allow_inf_nan=False)
    probabilities: SkipJsonSchema[dict[int, Probability] | None] = None
    level: SkipJsonSchema[int | None] = Field(default=None, ge=0, strict=True)
    options: ClassVar[tuple] = ()

    @classmethod
    def __class_getitem__(cls, options):
        if not isinstance(options, tuple) or len(options) < 2 or any(not isinstance(v, str) for v in options):
            raise ValueError("Score requires at least two ordered level descriptions, e.g. Score['poor', 'excellent'].")
        return _score_type(options)

    @classmethod
    def description(cls):
        return (
            f"Continuous score from 0 to {len(cls.options) - 1}; intermediate values are allowed. "
            "Rubric (level index, description): " + json.dumps(list(enumerate(cls.options))) + "."
        )

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        if source is float:
            # Pydantic calls this hook when this configured type is Annotated metadata.
            if not cls.options:
                raise ValueError("A native Score requires a configured rubric.")
            return core_schema.float_schema(ge=0, le=len(cls.options) - 1, allow_inf_nan=False)
        return handler(source)

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        result = handler(schema)
        if schema["type"] == "float":
            result["description"] = cls.description()
        return result

    def __float__(self):
        return self.value


@experimental
class Choice(_Decision):
    """A typed option value with confidence and optional provider probabilities.

    Declare ``Choice[("billing", "Payment issue"), ("technical", "Product bug")]``.
    Option values may be strings, integers, booleans, or None. Values retain
    their Python types; provider probability keys are their string labels.
    """

    value: str | int | bool | None
    probabilities: SkipJsonSchema[dict[str, Probability] | None] = None
    options: ClassVar[tuple] = ()

    @classmethod
    def __class_getitem__(cls, options):
        options = _option_pairs(options)
        values = [value for value, _ in options]
        if any(type(v) not in (str, int, bool, type(None)) for v in values):
            raise ValueError("Choice values must be strings, integers, booleans, or None.")
        if len({str(v) for v in values}) != len(values):
            raise ValueError("Choice values must have distinct string labels (e.g. 1 and '1' are ambiguous).")
        # Tuple equality conflates True with 1 and False with 0. Include the
        # literal types in the cache key so separate declarations stay distinct.
        return _choice_type(tuple((type(value), value, desc) for value, desc in options))

    @classmethod
    def description(cls):
        return "Select exactly one value. Options: " + json.dumps(cls.options) + "."


def _option_pairs(options):
    # Subscription with one pair passes the pair itself, rather than a tuple of pairs.
    if isinstance(options, tuple) and len(options) == 2 and isinstance(options[1], str):
        options = (options,)
    if (
        not isinstance(options, tuple)
        or not options
        or any(not isinstance(pair, tuple) or len(pair) != 2 or not isinstance(pair[1], str) for pair in options)
    ):
        raise ValueError("Declare options as (value, description) pairs.")
    return options


@lru_cache(maxsize=256)
def _score_type(options):
    result = create_model(
        f"Score[{', '.join(repr(description) for description in options)}]",
        __base__=Score,
        value=(float, Field(ge=0, le=len(options) - 1, allow_inf_nan=False)),
    )
    result.options = options
    return result


@lru_cache(maxsize=256)
def _choice_type(typed_options):
    options = tuple((value, desc) for _, value, desc in typed_options)
    result = create_model(
        f"Choice[{', '.join(repr(pair) for pair in options)}]",
        __base__=Choice,
        value=(Literal[tuple(value for value, _ in options)], ...),
    )
    result.options = options
    return result


def decision_type(field):
    """Resolve a supported field to its rich type without changing its annotation."""
    annotation = field.annotation
    if annotation is bool:
        return Noul
    if annotation is float:
        return next((m for m in field.metadata if isinstance(m, type) and issubclass(m, Score) and m.options), None)
    if get_origin(annotation) is Literal:
        return Choice[tuple((value, "") for value in get_args(annotation))]
    if isinstance(annotation, type) and issubclass(annotation, _Decision):
        if annotation is Noul or getattr(annotation, "options", ()):
            return annotation
    return None
