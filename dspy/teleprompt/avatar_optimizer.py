import dspy

from tqdm import tqdm
from copy import deepcopy
from random import sample
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Tuple, Optional

from dspy.teleprompt import Teleprompter
from dspy.predict.avatar import ActionOutput


DEFAULT_MAX_EXAMPLES = 10


class EvalResult(BaseModel):
    example: dict
    score: float
    actions: Optional[List[ActionOutput]] = None


class Comparator(dspy.Signature):
    """After executing the given actions on user inputs using the given instruction, some inputs have yielded good, results, while others have not. I'll provide you the inputs along with their, corresponding evaluation metrics:

Task:
(1) Firstly, identify and contrast the patterns of inputs that have achieved good results with those that have not.
(2) Then, review the computational logic for any inconsistencies in the previous actions.
(3) Lastly, specify the modification in tools used that can lead to improved performance on the negative inputs."""

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
        desc="Positive inputs along with their score on a evaluation metric and actions taken",
    )
    neg_input_with_metrics: List[EvalResult] = dspy.InputField(
        prefix="Negative Inputs: ",
        desc="Negative inputs along with their score on a evaluation metric and actions taken",
    )
    feedback: str = dspy.OutputField(
        prefix="Feedback: ",
        desc="Feedback for the actor to improve the performance of negative inputs",
    )


class AvatarOptimizer(Teleprompter):
    def __init__(
        self,
        metric: Callable,
        init_temperature: float = 1.4,
        max_iters: int = 10,
        lower_bound: int = 0,
        upper_bound: int = 1,
        max_positive_inputs: int = None,
        max_negative_inputs: int = None,
        optimize_for: str = "max",
    ):
        assert metric is not None, "`metric` argument cannot be None. Please provide a metric function."
        self.metric = metric
        self.optimize_for = optimize_for

        self.max_iters = max_iters
        self.init_temperature = init_temperature

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.max_positive_inputs = max_positive_inputs or DEFAULT_MAX_EXAMPLES
        self.max_negative_inputs = max_negative_inputs or DEFAULT_MAX_EXAMPLES

        self.comparator = dspy.TypedPredictor(Comparator)
    

    def process_example(self, actor, example, return_outputs):
        actor = deepcopy(actor)

        try:
            prediction = actor(**example.inputs().toDict())
            score = metric(example, prediction)

            match return_outputs:
                case True:
                    return example, prediction, score
                case False:
                    return score

        except Exception as e:
            print(e)
            
            match return_outputs:
                case True:
                    return example, None, 0
                case False:
                    return 0


    def multi_thread_executor(self, devset, actor, return_outputs=False, num_threads=60):
        total_score = 0
        total_examples = len(devset)
        results = []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self.process_example, actor, example, return_outputs) for example in devset]
            
            for future in tqdm(futures, total=total_examples, desc="Processing examples"):
                result = future.result()
                if return_outputs:
                    example, prediction, score = result
                    total_score += score
                    results.append((example, prediction, score))
                else:
                    total_score += result
        
        avg_metric = total_score / total_examples
        
        if return_outputs:
            return avg_metric, results
        else:
            return avg_metric


    def _get_pos_neg_results(
        self, 
        actor: dspy.Module, 
        trainset: List[dspy.Example]
    ) -> Tuple[List[EvalResult], List[EvalResult]]:
        pos_inputs = []
        neg_inputs = []
        
        _, results = self.multi_thread_executor(trainset, actor, return_outputs=True)

        for example, prediction, score in results:
            if score >= self.upper_bound:
                pos_inputs.append(
                    EvalResult(
                        example=example.inputs().toDict(),
                        score=score,
                        actions=prediction.actions if prediction else None
                    )
                )
            elif score <= self.lower_bound:
                neg_inputs.append(
                    EvalResult(
                        example=example.inputs().toDict(),
                        score=score,
                        actions=prediction.actions if prediction else None
                    )
                )

        if len(pos_inputs) == 0:
            raise ValueError("No positive examples found, try lowering the upper_bound or providing more training data")
        if len(neg_inputs) == 0:
            raise ValueError("No negative examples found, try raising the lower_bound or providing more training data")
        
        return (pos_inputs, neg_inputs)
    

    def compile(self, student, *, trainset):
        actor = deepcopy(student)
        
        best_feedback = None
        
        best_score = self.multi_thread_executor(trainset, actor)
        print(f"Average Unoptimized score: {best_score}")
        
        for i in range(self.max_iters):
            print(20*'=')
            print(f"Iteration {i+1}/{self.max_iters}")

            pos_inputs, neg_inputs = self._get_pos_neg_results(actor, trainset)
            print(f"Positive examples: {len(pos_inputs)}")
            print(f"Negative examples: {len(neg_inputs)}")
            print(f"Sampling {self.max_positive_inputs} positive examples and {self.max_negative_inputs} negative examples")

            if self.max_positive_inputs and len(pos_inputs) > self.max_positive_inputs:
                pos_inputs = sample(pos_inputs, self.max_positive_inputs)

            if self.max_negative_inputs and len(neg_inputs) > self.max_negative_inputs:
                neg_inputs = sample(neg_inputs, self.max_negative_inputs)

            feedback = self.comparator(
                instruction=actor.signature.instructions,
                actions=[str(tool) for tool in actor.tools],
                pos_input_with_metrics=pos_inputs,
                neg_input_with_metrics=neg_inputs
            ).feedback

            print(f"\n\nFeedback for iteration {i+1}: {feedback}\n\n")

            if actor.feedback is None:
                actor._add_feedback_field()
                actor.feedback = feedback
                best_feedback = feedback
            else:
                if self.optimize_for == "max" and self.metric(feedback) > self.metric(actor.feedback):
                    actor.feedback = feedback
                    best_feedback = feedback
                elif self.optimize_for == "min" and self.metric(feedback) < self.metric(actor.feedback):
                    actor.feedback = feedback
                    best_feedback = feedback

            avg_score = self.multi_thread_executor(trainset, actor)

            print(f"Average Score: {avg_score}")
        
        print(f"Best Feedback: {best_feedback}")

        return actor
            