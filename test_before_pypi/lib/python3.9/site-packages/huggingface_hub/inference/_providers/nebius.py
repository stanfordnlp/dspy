import base64
from typing import Any, Dict, Optional, Union

from huggingface_hub.inference._common import _as_dict
from huggingface_hub.inference._providers._common import (
    BaseConversationalTask,
    BaseTextGenerationTask,
    TaskProviderHelper,
    filter_none,
)


class NebiusTextGenerationTask(BaseTextGenerationTask):
    def __init__(self):
        super().__init__(provider="nebius", base_url="https://api.studio.nebius.ai")


class NebiusConversationalTask(BaseConversationalTask):
    def __init__(self):
        super().__init__(provider="nebius", base_url="https://api.studio.nebius.ai")


class NebiusTextToImageTask(TaskProviderHelper):
    def __init__(self):
        super().__init__(task="text-to-image", provider="nebius", base_url="https://api.studio.nebius.ai")

    def _prepare_route(self, mapped_model: str) -> str:
        return "/v1/images/generations"

    def _prepare_payload_as_dict(self, inputs: Any, parameters: Dict, mapped_model: str) -> Optional[Dict]:
        parameters = filter_none(parameters)
        if "guidance_scale" in parameters:
            parameters.pop("guidance_scale")
        if parameters.get("response_format") not in ("b64_json", "url"):
            parameters["response_format"] = "b64_json"

        return {"prompt": inputs, **parameters, "model": mapped_model}

    def get_response(self, response: Union[bytes, Dict]) -> Any:
        response_dict = _as_dict(response)
        return base64.b64decode(response_dict["data"][0]["b64_json"])
