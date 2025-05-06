from dspy.predict.predict import Predict
from dspy.primitives.program import Module
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
                f"reasoning_attempt_{idx+1}",
                InputField(
                    prefix=f"Student Attempt #{idx+1}:",
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

    def forward(self, completions, **kwargs):
        attempts = []

        for c in completions:
            rationale = c.get("rationale", c.get("reasoning")).strip().split("\n")[0].strip()
            answer = str(c[self.last_key]).strip().split("\n")[0].strip()
            attempts.append(
                f"«I'm trying to {rationale} I'm not sure but my prediction is {answer}»",
            )

        assert (
            len(attempts) == self.M
        ), f"The number of attempts ({len(attempts)}) doesn't match the expected number M ({self.M}). Please set the correct value for M when initializing MultiChainComparison."

        kwargs = {
            **{f"reasoning_attempt_{idx+1}": attempt for idx, attempt in enumerate(attempts)},
            **kwargs,
        }
        return self.predict(**kwargs)
