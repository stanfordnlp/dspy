"""Decision values shared by System One predictors and generative LMs."""

import json
from functools import lru_cache
from typing import Annotated, ClassVar, Literal, get_args, get_origin

from pydantic import ConfigDict, Field, JsonValue, TypeAdapter, create_model, model_serializer
from pydantic.json_schema import SkipJsonSchema
from pydantic_core import core_schema

from dspy.adapters.types.base_type import Type
from dspy.utils.annotation import experimental

Probability = Annotated[float, Field(ge=0, le=1)]
_JSON_ADAPTER = TypeAdapter(JsonValue)


class _Decision(Type):
    model_config = ConfigDict(extra="forbid")
    _criteria_json: ClassVar[str] = "null"

    confidence: Probability = Field(description="Confidence in the value, from 0 to 1.")

    @classmethod
    def criteria(cls):
        """Return an independent copy of the type's declared criteria."""
        return json.loads(cls._criteria_json)

    @model_serializer(mode="wrap")
    def serialize_model(self, handler):
        # Unlike media types, decisions are ordinary structured JSON, including
        # provider evidence when present. Missing evidence is not shown as null.
        return {key: value for key, value in handler(self).items() if value is not None or key == "value"}


@experimental
class Noul(_Decision):
    """A Boolean value with confidence and optional provider true-probability.

    Optionally declare ``Noul[(True, "blocked"), (False, "usable")]`` with one
    or both outcome descriptions. ``Annotated[bool, Noul[...]]`` retains these
    criteria while returning a native bool. Thresholds belong to Predict,
    not the type, and apply with both generative and decision backends.

    Predict obtains true-probability from the backend and derives value using
    its per-field threshold (default 0.5). Confidence is
    ``abs(p - threshold) / max(threshold, 1 - threshold)``:
    a distance from the decision boundary, not a calibrated probability.
    """

    value: bool = Field(strict=True)
    probability: SkipJsonSchema[Probability | None] = None

    @classmethod
    def __class_getitem__(cls, options):
        options = _option_pairs(options)
        if any(type(value) is not bool for value, _ in options) or len(dict(options)) != len(options):
            raise ValueError("Noul criteria require distinct True/False values with JSON descriptions.")
        return _noul_type(tuple((value, _description_json(desc)) for value, desc in sorted(options, reverse=True)))

    @classmethod
    def description(cls):
        description = "A Boolean decision."
        if cls.criteria():
            description += " Criteria: " + cls._criteria_json + "."
        return description

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return core_schema.bool_schema() if source is bool else handler(source)

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        result = handler(schema)
        result["description"] = cls.description()
        return result

    def __bool__(self):
        return self.value


@experimental
class Score(_Decision):
    """A continuous score with a declared rubric and confidence.

    Declare ``Score["poor", "fair", "excellent"]`` in increasing order.
    Predict obtains a distribution and confidence from the backend and
    averages the level indices using those probabilities.
    ``probabilities`` retains the raw distribution, keyed by rubric index.
    ``level`` is the zero-based ordinal selected by the predictor's cuts on the mean
    level index. It does not change the continuous value or provider confidence.
    The backend does not generate the derived value or level independently.

    Use the configured Score as the field annotation. Access ``.value`` or
    call ``float(result)`` to obtain the continuous numeric value.
    """

    value: float = Field(allow_inf_nan=False)
    probabilities: SkipJsonSchema[dict[int, Probability] | None] = None
    level: SkipJsonSchema[int | None] = Field(default=None, ge=0, strict=True)

    @classmethod
    def __class_getitem__(cls, options):
        if not isinstance(options, tuple) or len(options) < 2:
            raise ValueError("Score requires at least two ordered level descriptions, e.g. Score['poor', 'excellent'].")
        return _score_type(tuple(_description_json(desc) for desc in options))

    @classmethod
    def description(cls):
        criteria = cls.criteria() or []
        return (
            f"Continuous score from 0 to {len(criteria) - 1}; intermediate values are allowed. "
            "Rubric (level index, description): " + json.dumps(list(enumerate(criteria))) + "."
        )

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        if source is float:
            raise ValueError("Use Score[...] directly as the field type, not Annotated[float, Score[...]].")
        return handler(source)

    def __float__(self):
        return self.value


@experimental
class Choice(_Decision):
    """A typed option value with confidence and optional provider probabilities.

    Declare ``Choice[("billing", "Payment issue"), ("technical", "Product bug")]``.
    Option values may be strings, integers, booleans, or None. Values retain
    their Python types; provider probability keys are their string labels.
    ``Annotated[Literal[...], Choice[...]]`` supplies criteria and enables
    evidence decoding while returning a native member. Both declarations must
    contain the same typed values. Bare ``Choice`` metadata uses the Literal's values.
    """

    value: str | int | bool | None
    probabilities: SkipJsonSchema[dict[str, Probability] | None] = None

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
        return _choice_type(tuple((type(value), value, _description_json(desc)) for value, desc in options))

    @classmethod
    def description(cls):
        criteria = cls.criteria()
        if criteria is None:
            return "Select exactly one of the declared Literal values."
        options = [(v, criteria[str(v)]) for v in get_args(cls.model_fields["value"].annotation)]
        return "Select exactly one value. Options: " + json.dumps(options) + "."

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        if get_origin(source) is Literal:
            values = get_args(source)
            declared = get_args(cls.model_fields["value"].annotation)
            if cls.criteria() is not None and {(type(v), v) for v in values} != {(type(v), v) for v in declared}:
                raise ValueError("Choice criteria must match the Literal members, including their Python types.")
        return handler(source)

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        result = handler(schema)
        result["description"] = cls.description()
        return result


def _option_pairs(options):
    # Subscription with one pair passes the pair itself, rather than a tuple of pairs.
    if isinstance(options, tuple) and len(options) == 2 and not isinstance(options[0], tuple):
        options = (options,)
    if (
        not isinstance(options, tuple)
        or not options
        or any(not isinstance(pair, tuple) or len(pair) != 2 for pair in options)
    ):
        raise ValueError("Declare options as (value, description) pairs.")
    return options


def _description_json(description):
    """Validate and snapshot descriptions as hashable type-cache keys."""
    if not isinstance(description, (str, dict, list, type(None))):
        raise ValueError("Decision descriptions must be JSON strings, objects, arrays, or null.")
    _JSON_ADAPTER.validate_python(description, strict=True)
    return json.dumps(description, allow_nan=False)


@lru_cache(maxsize=256)
def _noul_type(options):
    options = tuple((value, json.loads(desc)) for value, desc in options)
    result = create_model(f"Noul[{', '.join(repr(pair) for pair in options)}]", __base__=Noul)
    result._criteria_json = json.dumps({str(v).lower(): desc for v, desc in options})
    return result


@lru_cache(maxsize=256)
def _score_type(options):
    options = tuple(json.loads(desc) for desc in options)
    result = create_model(
        f"Score[{', '.join(repr(description) for description in options)}]",
        __base__=Score,
        value=(float, Field(ge=0, le=len(options) - 1, allow_inf_nan=False)),
        level=(SkipJsonSchema[int | None], Field(default=None, ge=0, le=len(options) - 1, strict=True)),
    )
    result._criteria_json = json.dumps(options)
    return result


@lru_cache(maxsize=256)
def _choice_type(typed_options):
    options = tuple((value, json.loads(desc)) for _, value, desc in typed_options)
    result = create_model(
        f"Choice[{', '.join(repr(pair) for pair in options)}]",
        __base__=Choice,
        value=(Literal[tuple(value for value, _ in options)], ...),
    )
    result._criteria_json = json.dumps({str(v): desc for v, desc in options})
    return result


def decision_type(field):
    """Resolve a supported field to its rich type without changing its annotation."""
    annotation = field.annotation
    if annotation is bool:
        return next((m for m in field.metadata if isinstance(m, type) and issubclass(m, Noul)), Noul)
    if get_origin(annotation) is Literal:
        configured = next((m for m in field.metadata if isinstance(m, type) and issubclass(m, Choice)), None)
        if configured is not None and configured.criteria() is not None:
            return configured
        return Choice[tuple((value, "") for value in get_args(annotation))]
    if isinstance(annotation, type) and issubclass(annotation, _Decision):
        if issubclass(annotation, Noul) or annotation.criteria() is not None:
            return annotation
    return None
