from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.signatures import InputField, OutputField
from dspy.signatures.signature import ensure_signature


class MultiChainComparison(Module):
    """Multi-chain comparison module for aggregating multiple reasoning attempts.

    This module takes multiple candidate completions (e.g., from chain-of-thought
    prompting with different samples) and produces a refined answer by comparing
    and synthesizing the different reasoning attempts.

    The module works by:
    1. Extracting rationale and answer from each completion attempt
    2. Presenting all attempts to the LM with a comparison prompt
    3. Generating a refined "accurate reasoning" that synthesizes the best parts

    Attributes:
        M: Number of reasoning attempts to compare.
        last_key: The name of the final output field from the signature.
        predict: The underlying Predict module with the comparison signature.

    Example:
        ```python
        import dspy

        # Define a signature for the task
        class QA(dspy.Signature):
            question = dspy.InputField()
            answer = dspy.OutputField()

        # Create multi-chain comparison with 3 attempts
        mcc = MultiChainComparison(QA, M=3)

        # Generate multiple completions (e.g., using ChainOfThought)
        cot = dspy.ChainOfThought(QA)
        completions = [cot(question="What is 2+2?") for _ in range(3)]

        # Compare and synthesize
        result = mcc(completions, question="What is 2+2?")
        ```
    """

    def __init__(self, signature, M=3, temperature=0.7, **config):  # noqa: N803
        """Initialize the multi-chain comparison module.

        Args:
            signature: The DSPy signature defining input/output fields.
                The signature should have at least one output field.
            M: Number of reasoning attempts to compare. Must match the
                number of completions passed to forward(). Defaults to 3.
            temperature: Temperature for the comparison LM call.
                Higher values increase diversity. Defaults to 0.7.
            **config: Additional configuration passed to the underlying
                Predict module.
        """
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
        """Compare multiple completions and generate a refined answer.

        Args:
            completions: List of completion objects from previous predictions.
                Each completion should have either a 'rationale' or 'reasoning'
                key, plus the output field defined in the signature.
                Length must equal M.
            **kwargs: Input field values for the signature (e.g., question="...").

        Returns:
            dspy.Prediction containing the refined rationale and answer.

        Raises:
            AssertionError: If len(completions) != M.
        """
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
