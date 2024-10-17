"""
This file will reside in MLflow repository. Just implementing inside DSPy for making POC easier.
"""
from typing import Any, Dict

import dspy
from dsp.modules.lm import LM as DSPLM
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback

import mlflow

class MlflowCallback(BaseCallback):

    def __init__(self):
        import mlflow

        self._client = mlflow.MlflowClient()
        self._call_id_to_span = {}


    def on_start(
        self,
        call_id: str,
        function_name: str,
        instance: Any,
        inputs: Dict[str, Any],
        start_time: float
    ):
        if parent_call_id := ACTIVE_CALL_ID.get():
            parent_span = self._call_id_to_span.get(parent_call_id)

            if not parent_span:
                print("Something went wrong")
        elif active_span := mlflow.get_current_active_span():
            parent_span = active_span
        else:
            parent_span = None

        span_name = f"{instance.__class__.__name__}.{function_name}"
        span_type = self._get_span_type(instance)
        attributes = self._get_attributes(instance)

        common_params = dict(
            name=span_name,
            inputs=inputs,
            attributes=attributes,
            span_type=span_type,
            start_time_ns=int(start_time * 1e9),
        )

        if parent_span:
            span = self._client.start_span(
                request_id=parent_span.request_id,
                parent_id=parent_span.span_id,
                **common_params
            )
        else:
            span = self._client.start_trace(**common_params)

        self._call_id_to_span[call_id] = span

    def on_success(
        self,
        call_id: str,
        outputs: Dict[str, Any],
        end_time: float
    ):
        from mlflow.entities import SpanStatusCode

        span = self._call_id_to_span.pop(call_id)

        if isinstance(outputs, dspy.Prediction):
            outputs = {k: outputs.get(k) for k in outputs.keys()}

        common_params = dict(
            request_id=span.request_id,
            outputs=outputs,
            status=SpanStatusCode.OK,
            end_time_ns=int(end_time * 1e9),
        )

        if span.parent_id:
            self._client.end_span(span_id=span.span_id, **common_params)
        else:
            self._client.end_trace(**common_params)


    def on_failure(
        self,
        call_id: str,
        exception: Exception,
        end_time: float
    ):
        from mlflow.entities import SpanEvent, SpanStatusCode

        span = self._call_id_to_span.pop(call_id)
        span.add_event(SpanEvent.from_exception(exception))

        common_params = dict(
            request_id=span.request_id,
            status=SpanStatusCode.ERROR,
            end_time_ns=int(end_time * 1e9),
        )

        if span.parent_id:
            self._client.end_span(span_id=span.span_id, **common_params)
        else:
            self._client.end_trace(**common_params)


    def _get_span_type(self, instance):
        from mlflow.entities import SpanType

        if isinstance(instance, (dspy.LM, DSPLM)):
            if getattr(instance, "model_type", None) == "chat":
                return SpanType.CHAT_MODEL
            else:
                return SpanType.LLM
        elif isinstance(instance, dspy.Retrieve):
            return SpanType.RETRIEVER
        elif isinstance(instance, dspy.ReAct):
            return SpanType.AGENT
        elif isinstance(instance, dspy.Predict):
            return SpanType.LLM
        elif isinstance(instance, dspy.Adapter):
            return SpanType.PARSER
        elif isinstance(instance, dspy.Module):
            return SpanType.CHAIN
        else:
            return SpanType.UNKNOWN

    def _get_attributes(self, instance) -> Dict[str, Any]:
        attributes = {}
        if isinstance(instance, dspy.LM):
            attributes.update(instance.kwargs)
            attributes["model"] = instance.model
            attributes["model_type"] = instance.model_type
            attributes["cache"] = instance.cache
        elif isinstance(instance, DSPLM):
            attributes.update(instance.kwargs)
            attributes["provider"] = instance.provider
        elif isinstance(instance, dspy.Predict):
            attributes["signature"] = instance.signature.signature
        elif isinstance(instance, dspy.ChainOfThought):
            attributes["signature"] = instance.signature.signature
            attributes["extended_signature"] = instance.extended_signature.signature
        return attributes