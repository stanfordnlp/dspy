from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.signatures import InputField, OutputField
from dspy.signatures.signature import ensure_signature


class MultiChainComparison(Module):
    """Ensembles multiple reasoning chains and selects a final answer via comparison.

    This module implements the Multi-Chain Comparison technique where ``M`` separate
    reasoning attempts are generated, their rationales and answers extracted, and
    then fed to an internal ``Predict`` module to synthesize the best final response.

    The signature is augmented with ``M`` input fields for reasoning attempts and
    a prepended rationale output field that prompts holistic comparison.

    Args:
        signature: DSPy signature describing the inputs/outputs for prediction.
        M: Number of parallel reasoning attempts to compare. Defaults to 3.
        temperature: Sampling temperature passed to the underlying predictor.
            Defaults to 0.7.
        **config: Additional configuration forwarded to the underlying ``Predict``
            module.

    Example:
        Compare multiple chain-of-thought reasoning attempts:

        ```python
        import dspy

        # First generate M completions using ChainOfThought with n=M
        cot = dspy.ChainOfThought("question -> answer", n=3)
        completions = cot(question="What causes seasons on Earth?").completions

        # Then use MultiChainComparison to select the best answer
        mc = dspy.MultiChainComparison("question -> answer", M=3)
        result = mc(completions, question="What causes seasons on Earth?")
        print(result.answer)
        ```
    """

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
