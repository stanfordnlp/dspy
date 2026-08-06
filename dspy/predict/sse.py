from typing import Any
import dspy
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature, ensure_signature

class SimpleSelfEvolve(Module):
    def __init__(
        self,
        judge_signature: str | type[Signature],
        improve_signature: str | type[Signature],
        num_cycles: int = 3, 
        lm_judge: dspy.LM = None,
        lm_judges: list[dspy.LM] | list[tuple[dspy.LM, float]] = None,
        **config: dict[str, Any],
    ):
        """
        A minimal module that evolves responses through self-judgment and improvement cycles.

        Args:
            judge_signature: Signature for judging responses
            improve_signature: Signature for improving responses
            num_cycles: Number of evolution cycles
            lm_judge: Single judge LM to use
            lm_judges: List of judge LMs or list of (LM, probability) tuples
            **config: Configuration for predictors
        Example:
            ```python
            import dspy
            dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))
            
            # Judge Signature
            class JudgeResponse(dspy.Signature):
                question = dspy.InputField(desc="The original question or prompt")
                response = dspy.InputField(desc="The response to criticise and suggest improvements")
                previous_feedback = dspy.InputField(desc="Previous judgment, new feedback should differ new perspective", default=None)
                
                feedback = dspy.OutputField(desc="Specific instructions on how to improve the response: what to add, keep, edit, or expand on")

            # Improve signature
            class ImproveResponse(dspy.Signature):
                question = dspy.InputField(desc="The original question or prompt")
                previous_response = dspy.InputField(desc="Use feedback to improve on this response", default=None)
                feedback = dspy.InputField(desc="Feedback on what to keep, add, edit and new ideas to expand on", default=None)
                
                response = dspy.OutputField(desc="Original response based on question or an improved version of the previous_response based on question and feedback")

            evolve = SimpleSelfEvolve(
                judge_signature=JudgeResponse,
                improve_signature=ImproveResponse,
                cache=False,
                num_cycles=3
            )
            ```
        """
        super().__init__()
        self.num_cycles = num_cycles
        
        judge_signature = ensure_signature(judge_signature)
        if "previous_feedback" in judge_signature.input_fields.keys():
            judge_signature = judge_signature.append(
                name="previous_feedback", 
                field=dspy.InputField(desc="The previous response that needs improvement")
            )
        self.judge = dspy.Predict(judge_signature, **config)
        
        improve_signature = ensure_signature(improve_signature)
        if "previous_response" in improve_signature.input_fields.keys():
            improve_signature = improve_signature.append(
                name="previous_response", 
                field=dspy.InputField(desc="The previous response that needs improvement")
            )
        self.improve = dspy.Predict(improve_signature, **config)
        
        self.lm_judge = lm_judge
        self.lm_judges = None
        self.lm_judges_probs = None
        
        if lm_judges and not lm_judge:
            if all(isinstance(item, tuple) and len(item) == 2 for item in lm_judges):
                lms, probs = zip(*lm_judges)
                self.lm_judges = list(lms)
                self.lm_judges_probs = list(probs)
                
            elif all(isinstance(item, dspy.LM) for item in lm_judges):
                self.lm_judges = lm_judges
                self.lm_judges_probs = None
                
            else:
                raise TypeError(
                    "lm_judges must be either a list of dspy.LM or a list of (dspy.LM, float) tuples"
                )

    def get_judge(self):
        import random
        if self.lm_judges:
            if self.lm_judges_probs:
                return random.choices(self.lm_judges, weights=self.lm_judges_probs, k=1)[0]
            else:
                return random.choice(self.lm_judges)
        elif self.lm_judge:
            return self.lm_judge
        else:
            return None

    def forward(self, **kwargs):
        response = self.improve(**kwargs)

        for cycle in range(self.num_cycles):
            kwargs.update(response)
            selected_lm = self.get_judge()
            if selected_lm:
                kwargs["lm"] = selected_lm
            
            judgment = self.judge(**kwargs)
            kwargs['previous_feedback'] = judgment.get('feedback', "")
            kwargs.update(judgment)
            kwargs["previous_response"] = response.response
            current_result = self.improve(**kwargs)
        return current_result

    async def aforward(self, **kwargs):
        current_result = await self.improve.acall(**kwargs)

        for _ in range(self.num_cycles):
            kwargs.update(response)
            selected_lm = self.get_judge()

            if selected_lm:
                judge_kwargs["lm"] = selected_lm

            kwargs['previous_feedback'] = judgment.get('feedback', "")
            judgment = await self.judge.acall(**judge_kwargs)
            kwargs.update(judgment)
            kwargs["previous_response"] = response.response
            current_result = await self.improve.acall(**kwargs)

        return current_result

    def update_config(self, **kwargs):
        self.judge.update_config(**kwargs)
        self.improve.update_config(**kwargs)

    def get_config(self):
        return self.judge.get_config()

    def __repr__(self):
        return f"{self.__class__.__name__}(num_cycles={self.num_cycles})"