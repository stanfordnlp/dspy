import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from huggingface_hub import constants
from huggingface_hub.inference._common import _b64_encode, _open_as_binary
from huggingface_hub.inference._providers._common import TaskProviderHelper, filter_none
from huggingface_hub.utils import build_hf_headers, get_session, get_token, hf_raise_for_status


class HFInferenceTask(TaskProviderHelper):
    """Base class for HF Inference API tasks."""

    def __init__(self, task: str):
        super().__init__(
            provider="hf-inference",
            base_url=constants.INFERENCE_PROXY_TEMPLATE.format(provider="hf-inference"),
            task=task,
        )

    def _prepare_api_key(self, api_key: Optional[str]) -> str:
        # special case: for HF Inference we allow not providing an API key
        return api_key or get_token()  # type: ignore[return-value]

    def _prepare_mapped_model(self, model: Optional[str]) -> str:
        if model is not None:
            return model
        model = _fetch_recommended_models().get(self.task)
        if model is None:
            raise ValueError(
                f"Task {self.task} has no recommended model for HF Inference. Please specify a model"
                " explicitly. Visit https://huggingface.co/tasks for more info."
            )
        return model

    def _prepare_url(self, api_key: str, mapped_model: str) -> str:
        # hf-inference provider can handle URLs (e.g. Inference Endpoints or TGI deployment)
        if mapped_model.startswith(("http://", "https://")):
            return mapped_model
        return (
            # Feature-extraction and sentence-similarity are the only cases where we handle models with several tasks.
            f"{self.base_url}/pipeline/{self.task}/{mapped_model}"
            if self.task in ("feature-extraction", "sentence-similarity")
            # Otherwise, we use the default endpoint
            else f"{self.base_url}/models/{mapped_model}"
        )

    def _prepare_payload_as_dict(self, inputs: Any, parameters: Dict, mapped_model: str) -> Optional[Dict]:
        if isinstance(inputs, bytes):
            raise ValueError(f"Unexpected binary input for task {self.task}.")
        if isinstance(inputs, Path):
            raise ValueError(f"Unexpected path input for task {self.task} (got {inputs})")
        return {"inputs": inputs, "parameters": filter_none(parameters)}


class HFInferenceBinaryInputTask(HFInferenceTask):
    def _prepare_payload_as_dict(self, inputs: Any, parameters: Dict, mapped_model: str) -> Optional[Dict]:
        return None

    def _prepare_payload_as_bytes(
        self, inputs: Any, parameters: Dict, mapped_model: str, extra_payload: Optional[Dict]
    ) -> Optional[bytes]:
        parameters = filter_none({k: v for k, v in parameters.items() if v is not None})
        extra_payload = extra_payload or {}
        has_parameters = len(parameters) > 0 or len(extra_payload) > 0

        # Raise if not a binary object or a local path or a URL.
        if not isinstance(inputs, (bytes, Path)) and not isinstance(inputs, str):
            raise ValueError(f"Expected binary inputs or a local path or a URL. Got {inputs}")

        # Send inputs as raw content when no parameters are provided
        if not has_parameters:
            with _open_as_binary(inputs) as data:
                data_as_bytes = data if isinstance(data, bytes) else data.read()
                return data_as_bytes

        # Otherwise encode as b64
        return json.dumps({"inputs": _b64_encode(inputs), "parameters": parameters, **extra_payload}).encode("utf-8")


class HFInferenceConversational(HFInferenceTask):
    def __init__(self):
        super().__init__("text-generation")

    def _prepare_payload_as_dict(self, inputs: Any, parameters: Dict, mapped_model: str) -> Optional[Dict]:
        payload_model = parameters.get("model") or mapped_model

        if payload_model is None or payload_model.startswith(("http://", "https://")):
            payload_model = "dummy"

        return {**filter_none(parameters), "model": payload_model, "messages": inputs}

    def _prepare_url(self, api_key: str, mapped_model: str) -> str:
        base_url = (
            mapped_model
            if mapped_model.startswith(("http://", "https://"))
            else f"{constants.INFERENCE_PROXY_TEMPLATE.format(provider='hf-inference')}/models/{mapped_model}"
        )
        return _build_chat_completion_url(base_url)


def _build_chat_completion_url(model_url: str) -> str:
    # Strip trailing /
    model_url = model_url.rstrip("/")

    # Append /chat/completions if not already present
    if model_url.endswith("/v1"):
        model_url += "/chat/completions"

    # Append /v1/chat/completions if not already present
    if not model_url.endswith("/chat/completions"):
        model_url += "/v1/chat/completions"

    return model_url


@lru_cache(maxsize=1)
def _fetch_recommended_models() -> Dict[str, Optional[str]]:
    response = get_session().get(f"{constants.ENDPOINT}/api/tasks", headers=build_hf_headers())
    hf_raise_for_status(response)
    return {task: next(iter(details["widgetModels"]), None) for task, details in response.json().items()}
