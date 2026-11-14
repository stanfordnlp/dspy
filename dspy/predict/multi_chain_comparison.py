from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.signatures import InputField, OutputField
from dspy.signatures.signature import ensure_signature


class MultiChainComparison(Module):
    def __init__(self, signature, M=3, temperature=0.7, **config):  # noqa: N803
        super().__init__()

        self.M = M
        signature = ensure_signature(signature)

        *_, self.last_key = signature.output_fields.keys()

        for idx in range(M):
            signature = signature.append(
                f"reasoning_attempt_{idx + 1}",
                InputField(
                    prefix=f"Student Attempt #{idx + 1}:",
                    desc="${reasoning attempt}",
                ),
            )

        signature = signature.prepend(
            "rationale",
            OutputField(
                prefix="Accurate Reasoning: Thank you everyone. Let's now holistically",
                desc="${corrected reasoning}",
            ),
        )

        self.predict = Predict(signature, temperature=temperature, **config)

    def _prepare_kwargs(self, completions, kwargs):
        if len(completions) != self.M:
            raise ValueError(
                f"The number of attempts ({len(completions)}) doesn't match the expected number M ({self.M}). "
                f"Please set the correct value for M when initializing MultiChainComparison."
            )

        attempts = []
        for c in completions:
            rationale_val = c.get("rationale") if hasattr(c, "get") else getattr(c, "rationale", None)
            if rationale_val is None:
                rationale_val = c.get("reasoning") if hasattr(c, "get") else getattr(c, "reasoning", None)
            rationale = str(rationale_val or "").strip().split("\n")[0].strip()

            ans_val = c.get(self.last_key) if hasattr(c, "get") else getattr(c, self.last_key, None)
            answer = str(ans_val or "").strip().split("\n")[0].strip()

            attempts.append(
                f"«I'm trying to {rationale} I'm not sure but my prediction is {answer}»"
                if rationale
                else f"«My prediction is {answer}»",
            )

        return {
            **{f"reasoning_attempt_{idx + 1}": attempt for idx, attempt in enumerate(attempts)},
            **kwargs,
        }

    def forward(self, completions, **kwargs):
        kwargs = self._prepare_kwargs(completions, kwargs)
        return self.predict(**kwargs)

    async def aforward(self, completions, **kwargs):
        kwargs = self._prepare_kwargs(completions, kwargs)
        return await self.predict.acall(**kwargs)
