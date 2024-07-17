from copy import deepcopy
from random import sample
from typing import Callable, List, Tuple

from pydantic import BaseModel
import dspy

from dspy.evaluate import Evaluate
from dspy.teleprompt import Teleprompter


class EvalResult(BaseModel):
    example: dspy.Example
    score: float


class Comparator(dspy.Signature):
    """After executing the given actions on user queries using the given instruction, some queries have yielded good, results, while others have not. I'll provide you the queries along with their, corresponding evaluation metrics:

Task:
(1) Firstly, identify and contrast the patterns of queries that have achieved good, results with those that have not.
(2) Then, review the computational logic for any inconsistencies in the previous, actions.
(3) Lastly, specify the modification that can lead to improved performance on the, negative queries. You should focus on capturing the high-level pattern of, the queries relevant to the knowledge base schema."""

    instruction: str = dspy.InputField(
        prefix="Instruction: ",
        desc="Instruction for the actor to execute the task",
    )
    actions: List[str] = dspy.InputField(
        prefix="Actions: ",
        desc="Actions actor can take to complete the task",
    )
    pos_input_with_metrics: List[EvalResult] = dspy.InputField(
        prefix="Positive Inputs: ",
        desc="Positive inputs along with their score on a evaluation metric",
    )
    neg_input_with_metrics: List[EvalResult] = dspy.InputField(
        prefix="Negative Inputs: ",
        desc="Negative inputs along with their score on a evaluation metric",
    )
    feedback: str = dspy.OutputField(
        prefix="Feedback: ",
        desc="Feedback for the actor to improve the performance of negative inputs",
    )


class AvatarOptimizer(Teleprompter):
    def __init__(
        self,
        prompt_model: dspy.LM = None,
        task_model: dspy.LM = None,
        metric: Callable = None,
        init_temperature: float = 1.4,
        max_iters: int = 10,
        lower_bound: int = 0,
        upper_bound: int = 1,
        max_positive_inputs: int = None,
        max_negative_inputs: int = None,
        metric_order: str = "max",
    ):
        self.prompt_model = prompt_model or dspy.settings.lm
        self.task_model = task_model or dspy.settings.lm

        assert metric is not None, "`metric` argument cannot be None. Please provide a metric function."
        self.metric = metric
        self.metric_order = metric_order

        self.max_iters = max_iters
        self.init_temperature = init_temperature

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.max_positive_inputs = max_positive_inputs
        self.max_negative_inputs = max_negative_inputs

        self.comparator = dspy.TypedPredictor(Comparator)


    def _get_pos_neg_results(
        self, 
        actor: dspy.Module, 
        trainset: List[dspy.Example]
    ) -> Tuple[List[EvalResult], List[EvalResult]]:
        pos_inputs = []
        neg_inputs = []
        
        _, results = self.evaluate(actor, trainset, return_outputs=True)

        for example, _, score in results:
            if score >= self.upper_bound:
                pos_inputs.append(
                    EvalResult(
                        example=example.inputs(),
                        score=score
                    )
                )
            elif score <= self.lower_bound:
                neg_inputs.append(
                    EvalResult(
                        example=example.inputs(),
                        score=score
                    )
                )
        if self.verbose:
            print(f"Positive examples: {len(pos_inputs)}")
            print(f"Negative examples: {len(neg_inputs)}")

        if len(pos_inputs) == 0:
            raise ValueError("No positive examples found, try lowering the upper_bound or providing more training data")
        if len(neg_inputs) == 0:
            raise ValueError("No negative examples found, try raising the lower_bound or providing more training data")
        
        return (pos_inputs, neg_inputs)


    def compile(self, student, *, trainset, eval_kwargs):
        actor = deepcopy(student)
        
        best_feedback = None
        self.evaluate = Evaluate(actor, metric=self.metric, **eval_kwargs)
        
        best_score = self.evaluate(actor, trainset)
        print(f"Average Unoptimized score: {best_score}")
        
        for i in range(self.max_iters):
            print(f"Iteration {i+1}/{self.max_iters}")

            pos_results, neg_results = self._get_pos_neg_results(actor, trainset)

            if self.max_positive_inputs and len(pos_results) > self.max_positive_inputs:
                pos_results = sample(pos_results, self.max_positive_inputs)

            if self.max_negative_inputs and len(neg_results) > self.max_negative_inputs:
                neg_results = sample(neg_results, self.max_negative_inputs)

            feedback = self.comparator(
                instruction=actor.goal,
                actions=[str(tool) for tool in actor.tools],
                pos_input_with_metrics=pos_results,
                neg_input_with_metrics=neg_results
            ).feedback

            if actor.feedback is None:
                actor._add_feedback_field()
                actor.feedback = feedback
                best_feedback = feedback
            else:
                if self.metric_order == "max" and self.metric(feedback) > self.metric(actor.feedback):
                    actor.feedback = feedback
                    best_feedback = feedback
                elif self.metric_order == "min" and self.metric(feedback) < self.metric(actor.feedback):
                    actor.feedback = feedback
                    best_feedback = feedback

            avg_score = self.evaluate(actor, trainset)

            if self.verbose:
                print(f"Average Score: {avg_score}")
        
        print(f"Best Feedback: {best_feedback}")

        return actor
            