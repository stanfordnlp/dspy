import inspect
from functools import partial
from types import MethodType

from dspy.predict.chain_of_thought import ChainOfThought
from dspy.primitives.program import Module
from dspy.refine.feedback import GenerateFeedback
from dspy.refine.utils import get_traces
from dspy.signatures.field import InputField


class Refine(Module):
    def __init__(self, module, metrics, metric_thresholds=None, max_iter=3):
        self.module = module.deepcopy()
        self.metrics = metrics
        self.metric_thresholds = metric_thresholds
        self.max_iter = max_iter

        self.metric_descriptions = [self._get_metric_description(metric) for metric in metrics]

        self._named_predicts = {name: predict for name, predict in self.module.named_predictors()}

    def _get_metric_description(self, metric):
        if hasattr(metric, "__repr__"):
            return str(metric)
        else:
            return inspect.getsource(metric.__class__)

    def _patch_predict_call_with_feedback(self, feedbacks):
        named_predicts = {}
        for name in feedbacks.keys():
            # Only patch the predict that has feedback.
            named_predicts[name] = self._named_predicts[name]

        predict_traces = get_traces(named_predicts)

        def forward_with_feedback(instance, dspy_refine_feedback, dspy_refine_last_trace, **kwargs):
            return instance.original_forward(
                **kwargs,
                dspy_refine_feedback=dspy_refine_feedback,
                dspy_refine_last_trace=dspy_refine_last_trace,
            )

        for name, predict in named_predicts.items():
            last_trace = predict_traces.get(name, None)
            if last_trace is None:
                continue
            # We trim out the last round's feedback and last_trace from the inputs to avoid too much nesting.
            if "dspy_refine_feedback" in last_trace["inputs"]:
                del last_trace["inputs"]["dspy_refine_feedback"]
            if "dspy_refine_last_trace" in last_trace["inputs"]:
                del last_trace["inputs"]["dspy_refine_last_trace"]

            feedback = feedbacks.get(name, None)
            if not hasattr(predict, "original_forward"):
                # If the predict has never been patched for refine calls, patch it.
                predict.original_signature = predict.signature
                predict.signature = predict.signature.prepend(
                    "dspy_refine_feedback",
                    InputField(desc="Improvement suggestion based on last try", type=str),
                ).prepend("dspy_refine_last_trace", InputField(desc="Trace of the last try", type=dict))

                # Save the original forward method before patching.
                predict.original_forward = predict.forward

            partial_forward = partial(
                forward_with_feedback, dspy_refine_feedback=feedback, dspy_refine_last_trace=last_trace
            )
            # Patch the `forward` method to the `forward_with_feedback` methd with partial values of feedback and
            # last_trace.
            predict.forward = MethodType(partial_forward, predict)

    def _undo_patch_predict_call_with_feedback(self, named_predicts):
        for _, predict in named_predicts.items():
            if hasattr(predict, "original_forward"):
                predict.forward = predict.original_forward
                predict.signature = predict.original_signature
                del predict.original_signature
                del predict.original_forward

    def _get_feedback_for_predicts(self, inputs, outputs):
        metric_descriptions = []
        metric_values = []
        for i, metric in enumerate(self.metrics):
            metric_value = metric(inputs, outputs)
            if self.metric_thresholds and metric_value < self.metric_thresholds[i]:
                metric_descriptions.append(self.metric_descriptions[i])
                metric_values.append(metric_value)

        if len(metric_descriptions) == 0:
            # All metric values are above the threshold, no need to refine.
            return {}

        feedback_program = ChainOfThought(GenerateFeedback)
        # Get feedback for each metric.
        feedbacks = feedback_program(
            metrics=metric_descriptions,
            metric_values=metric_values,
            module_inputs=inputs,
            module_outputs=outputs,
            source_code=inspect.getsource(self.module.__class__),
        ).feedback
        named_predicts = self._named_predicts

        predict_name_to_feedback = {}
        for name in named_predicts.keys():
            top_module_name = name.split(".")[0]
            if top_module_name in feedbacks:
                predict_name_to_feedback[name] = feedbacks[top_module_name]
            elif f"self.{top_module_name}" in feedbacks:
                predict_name_to_feedback[name] = feedbacks[f"self.{top_module_name}"]
        return predict_name_to_feedback

    def __call__(self, **kwargs):
        outputs = self.module(**kwargs)

        for i in range(self.max_iter):
            feedbacks = self._get_feedback_for_predicts(kwargs, outputs)

            if len(feedbacks) == 0:
                break
            self._patch_predict_call_with_feedback(feedbacks)

            outputs = self.module(**kwargs)

        named_predicts = {name: predict for name, predict in self.module.named_predictors()}
        self._undo_patch_predict_call_with_feedback(named_predicts)
        return outputs
