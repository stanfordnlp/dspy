import asyncio
import math
from collections import Counter
from typing import Literal, get_origin

import dspy
from dspy.primitives.module import Module
from dspy.signatures.signature import ensure_signature


class Classifier(Module):
    def __init__(self, signature, need_confidence=False, num_samples=10, classification_field=None, **config):
        """A module that classifies inputs, optionally computing statistical confidence from multiple samples.

        By default, runs a single prediction and returns the result directly. When
        need_confidence=True, runs the prediction num_samples times and computes
        confidence from the distribution of answers instead of asking the LM to
        hallucinate a confidence score.

        Args:
            signature: The signature of the module.
            need_confidence: If True, run num_samples predictions and return confidence
                metrics (agreement_rate, entropy, prediction_counts). Default False.
            num_samples: Number of times to run the prediction when need_confidence=True
                (default 10).
            classification_field: The output field to aggregate on. Auto-detected from
                Literal-typed fields if not specified.
            **config: Additional configuration passed to the underlying Predict module.
        """
        super().__init__()
        self.signature = ensure_signature(signature)
        self.need_confidence = need_confidence
        self.num_samples = num_samples
        self.classification_field = self._detect_classification_field(classification_field)
        self.predict = dspy.Predict(self.signature, **config)

    def _detect_classification_field(self, classification_field = None):
        """Resolve the classification field from Literal-typed output fields."""
        literal_fields = [
            name for name, field_info in self.signature.output_fields.items()
            if get_origin(field_info.annotation) is Literal
        ]
        if len(literal_fields) == 0:
            raise ValueError(
                "Classifier requires at least one Literal-typed output field. "
                "Example: `label: Literal['positive', 'negative'] = dspy.OutputField()`"
            )
        if len(literal_fields) == 1:
            if classification_field is None or classification_field == literal_fields[0]:
                return literal_fields[0]
            raise ValueError(
                f"classification_field='{classification_field}' does not match the only "
                f"Literal-typed output field '{literal_fields[0]}'."
            )
        # len(literal_fields) > 1
        if classification_field in literal_fields:
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

    def _aggregate_and_score(self, predictions):
        """Compute majority vote and confidence metrics from independent predictions."""
        field = self.classification_field
        values = [pred[field] for pred in predictions]

        # Majority vote
        counts = Counter(values)
        majority_value, majority_count = counts.most_common(1)[0]

        # Agreement rate: fraction of predictions matching the most common class
        agreement_rate = majority_count / len(values)

        # Normalized Shannon entropy (0 = perfect agreement, 1 = max disagreement)
        num_unique = len(counts)
        if num_unique <= 1:
            entropy = 0.0
        else:
            n = len(values)
            raw_entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())
            entropy = raw_entropy / math.log2(num_unique)

        # Build result from first prediction matching majority
        for pred in predictions:
            if pred[field] == majority_value:
                result = pred
                break

        result.agreement_rate = agreement_rate
        result.entropy = entropy
        result.prediction_counts = dict(counts)
        return result

    def _ensure_config(self, kwargs):
        """Extract config and set minimum temperature for diverse sampling.
        
        Explicitly sets n=1 to ensure compatibility with models that don't support
        n > 1, since Classifier achieves multiple samples via separate calls.
        """
        config = {**kwargs.pop("config", {})}
        # Explicitly set n=1 to override any LM-level n setting.
        # Classifier makes num_samples separate calls instead of using batch n.
        config["n"] = 1
        temperature = config.get("temperature")
        if temperature is not None and temperature < 0.15:
            config["temperature"] = 0.5
        return config, kwargs

    def forward(self, **kwargs):
        if not self.need_confidence:
            return self.predict(**kwargs)
        config, kwargs = self._ensure_config(kwargs)
        predictions = [
            self.predict(**kwargs, config={**config, "rollout_id": i})
            for i in range(self.num_samples)
        ]
        return self._aggregate_and_score(predictions)

    async def aforward(self, **kwargs):
        if not self.need_confidence:
            return await self.predict.acall(**kwargs)
        config, kwargs = self._ensure_config(kwargs)
        predictions = await asyncio.gather(
            *[
                self.predict.acall(**kwargs, config={**config, "rollout_id": i})
                for i in range(self.num_samples)
            ]
        )
        return self._aggregate_and_score(predictions)
