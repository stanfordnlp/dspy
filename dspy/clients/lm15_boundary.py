"""Typed direct calls using the exact lm15 objects bundled with DSPy.

Execution controls (response caching, retries and history) stay in DSPy. The
provider-neutral Request contains only what is sent to the model.
"""

import json

from dspy._vendor.lm15.providers.base import HttpResponse
from dspy._vendor.lm15.serde import part_to_dict
from dspy.clients.legacy_outputs import plain
from dspy.lm15 import OpenAIChatLM, OpenAILM, Request, response_from_openai_chat


class _NoTransport:
    def stream(self, request):
        raise RuntimeError("The typed conversion boundary must not perform network I/O")


def request_kwargs(request: Request, model_type: str) -> dict:
    """Serialize with lm15's dialect, never a second copy of its type system."""
    if "n" in (request.config.extensions or {}):
        raise ValueError("A typed Request produces one Response; do not put n in Config.extensions.")
    if model_type not in {"chat", "responses", "text"}:
        raise ValueError(f"Unsupported model_type: {model_type!r}")
    # Private dialect mapping is confined here because lm15 does not expose a
    # public outbound-body converter. No credentials are resolved or sent.
    dialect = OpenAIChatLM(api_key="conversion-only", transport=_NoTransport())
    if model_type == "responses":
        dialect = OpenAILM(api_key="conversion-only", transport=_NoTransport())
    data = dialect._payload(request, stream=False)
    data.pop("model", None)
    data.pop("stream", None)
    return data


def history_messages(request: Request) -> list[dict]:
    """A display snapshot; never read local files while recording history."""
    messages = []
    if request.system is not None:
        system = request.system
        messages.append({"role": "system", "content": system if isinstance(system, str)
                         else [part_to_dict(part) for part in system]})
    for message in request.messages:
        messages.append({"role": message.role, "content": message.text if message.text is not None
                         else [part_to_dict(part) for part in message.parts]})
    return messages


def response_value(response, model_type: str, request: Request):
    body = plain(response)
    if model_type == "responses":
        dialect = OpenAILM(api_key="conversion-only", transport=_NoTransport())
        return dialect.parse_response(request, HttpResponse(
            status=200, reason="OK", headers=[], body=json.dumps(body).encode(),
        ))
    if model_type == "text":
        body = dict(body)
        body["choices"] = [
            {**choice, "message": {"role": "assistant", "content": choice.get("text")}}
            for choice in body.get("choices", [])
        ]
    return response_from_openai_chat(body, model=request.model)
