import asyncio
import math
from collections import Counter
from typing import Literal, get_args, get_origin

import dspy
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature


class Classifier(Module):
    """A module that classifies inputs using Literal-typed output fields.

    The Classifier wraps ``dspy.Predict`` with a signature that has at least one
    ``Literal``-typed output field.  It supports two modes of operation:

    1. **Single prediction** (default):  Runs one prediction and returns the
       result directly -- identical to ``dspy.Predict``.

    2. **Confidence mode** (``need_confidence=True``):  Runs ``num_samples``
       independent predictions and computes statistical confidence from the
       distribution of answers:

       * ``agreement_rate`` -- fraction of samples matching the majority class.
       * ``entropy`` -- normalized Shannon entropy (0 = perfect agreement,
         1 = maximum disagreement).
       * ``prediction_counts`` -- raw count per class.

    The classification field is auto-detected from ``Literal``-typed output
    fields.  When the signature contains multiple ``Literal`` fields, the
    caller must pass ``classification_field`` to disambiguate.

    Example::

        class Sentiment(dspy.Signature):
            \"\"\"Classify the sentiment of a sentence.\"\"\"
            sentence: str = dspy.InputField()
            sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()

        classifier = dspy.Classifier(Sentiment, need_confidence=True, num_samples=10)
        result = classifier(sentence="This movie was great!")
        print(result.sentiment)           # "positive"
        print(result.agreement_rate)      # 0.9
        print(result.entropy)             # 0.12
        print(result.prediction_counts)   # {"positive": 9, "neutral": 1}
    """

    def __init__(
        self,
        signature,
        need_confidence: bool = False,
        num_samples: int = 10,
        classification_field: str | None = None,
        **config,
    ):
        """Initialize the Classifier module.

        Args:
            signature: A DSPy signature (class or string) with at least one
                ``Literal``-typed output field.
            need_confidence: If ``True``, run ``num_samples`` predictions and
                return statistical confidence metrics.  Default ``False``.
            num_samples: Number of independent predictions to run when
                ``need_confidence=True``.  Ignored otherwise.  Default 10.
            classification_field: Name of the ``Literal``-typed output field to
                aggregate on.  Auto-detected when the signature has exactly one
                such field.
            **config: Additional configuration forwarded to the underlying
                ``dspy.Predict`` module.
        """
        super().__init__()
        self.signature = ensure_signature(signature)
        self.need_confidence = need_confidence
        self.num_samples = num_samples
        self.classification_field = self._resolve_classification_field(classification_field)
        self._classes = self._extract_classes()
        self.predict = dspy.Predict(self.signature, **config)

    def _resolve_classification_field(self, classification_field: str | None = None) -> str:
        """Identify the Literal-typed output field to use for classification.

        Raises:
            ValueError: If no Literal-typed output field exists, if the
                explicit ``classification_field`` does not match any Literal
                field, or if multiple Literal fields exist and none is specified.
        """
        literal_fields = [
            name
            for name, field_info in self.signature.output_fields.items()
            if get_origin(field_info.annotation) is Literal
        ]

        if len(literal_fields) == 0:
            raise ValueError(
                "Classifier requires at least one Literal-typed output field in the "
                "signature.  Example: `label: Literal['positive', 'negative'] = "
                "dspy.OutputField()`"
            )

        if len(literal_fields) == 1:
            if classification_field is None or classification_field == literal_fields[0]:
                return literal_fields[0]
            raise ValueError(
                f"classification_field='{classification_field}' does not match the only "
                f"Literal-typed output field '{literal_fields[0]}'."
            )

        # Multiple Literal-typed fields -- user must disambiguate.
        if classification_field is not None and classification_field in literal_fields:
            return classification_field

        raise ValueError(
            f"Signature has multiple Literal-typed output fields: {literal_fields}. "
            + (
                f"classification_field='{classification_field}' is not among them. "
                if classification_field is not None
                else ""
            )
            + "Please specify which to use via `classification_field`."
        )

    def _extract_classes(self) -> tuple[str, ...]:
        """Return the allowed class labels from the Literal annotation."""
        annotation = self.signature.output_fields[self.classification_field].annotation
        return get_args(annotation)

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _aggregate_and_score(self, predictions):
        """Compute majority vote and confidence metrics from multiple predictions.

        Returns a ``dspy.Prediction`` augmented with ``agreement_rate``,
        ``entropy``, and ``prediction_counts``.
        """
        field = self.classification_field
        values = [pred[field] for pred in predictions]

        counts = Counter(values)
        majority_value, majority_count = counts.most_common(1)[0]

        # Agreement rate: fraction of predictions matching the majority class.
        agreement_rate = majority_count / len(values)

        # Normalized Shannon entropy.
        #   0.0 = perfect agreement (all samples return the same class)
        #   1.0 = maximum disagreement (uniform distribution over observed classes)
        num_unique = len(counts)
        if num_unique <= 1:
            entropy = 0.0
        else:
            n = len(values)
            raw_entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
            entropy = raw_entropy / math.log2(num_unique)

        # Use the first prediction whose classification matches the majority.
        for pred in predictions:
            if pred[field] == majority_value:
                result = pred
                break

        result.agreement_rate = agreement_rate
        result.entropy = entropy
        result.prediction_counts = dict(counts)
        return result

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _prepare_config(self, kwargs):
        """Extract the per-call config and ensure diverse sampling settings.

        When ``need_confidence=True``, Classifier makes ``num_samples`` separate
        calls rather than relying on the ``n`` parameter (which some providers
        do not support).  We therefore force ``n=1`` and bump very-low
        temperatures to 0.5 so that the repeated calls can produce varied
        outputs.

        Returns:
            A ``(config_dict, remaining_kwargs)`` tuple.
        """
        config = {**kwargs.pop("config", {})}
        # Force n=1 -- Classifier achieves multiple samples via separate calls.
        config["n"] = 1
        temperature = config.get("temperature")
        if temperature is not None and temperature < 0.15:
            config["temperature"] = 0.5
        return config, kwargs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, **kwargs):
        if not self.need_confidence:
            return self.predict(**kwargs)

        config, kwargs = self._prepare_config(kwargs)
        predictions = [
            self.predict(**kwargs, config={**config, "rollout_id": i})
            for i in range(self.num_samples)
        ]
        return self._aggregate_and_score(predictions)

    async def aforward(self, **kwargs):
        if not self.need_confidence:
            return await self.predict.acall(**kwargs)

        config, kwargs = self._prepare_config(kwargs)
        predictions = await asyncio.gather(
            *[
                self.predict.acall(**kwargs, config={**config, "rollout_id": i})
                for i in range(self.num_samples)
            ]
        )
        return self._aggregate_and_score(predictions)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def classes(self) -> tuple[str, ...]:
        """Return the allowed class labels derived from the Literal annotation."""
        return self._classes

    def __repr__(self) -> str:
        return (
            f"Classifier(classification_field={self.classification_field!r}, "
            f"classes={self._classes}, need_confidence={self.need_confidence}, "
            f"num_samples={self.num_samples})"
        )
