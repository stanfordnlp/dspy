"""DSPy execution metadata around unmodified lm15 responses."""

from dataclasses import dataclass, field

from dspy._vendor.lm15.serde import response_from_dict, response_to_dict
from dspy.clients.lm15_boundary import plain
from dspy.lm15 import Response, TextPart, ThinkingPart

CACHE_FORMAT = "dspy-lm15-result-v1"


class AttributeDict(dict):
    """Convenience outputs remain accessible by key and by attribute."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def model_dump(self, **kwargs):
        return plain(self)


def attributes(obj):
    if isinstance(obj, dict):
        return AttributeDict({key: attributes(item) for key, item in obj.items()})
    if isinstance(obj, list):
        return [attributes(item) for item in obj]
    return obj


def convenience_output(response: Response, *, logprobs=False):
    """The list-call view of one response: a string, or a dict with extra fields."""
    import json

    texts = [part.text for part in response.message.parts if isinstance(part, TextPart)]
    thinking = [part.text for part in response.message.parts if isinstance(part, ThinkingPart)]
    output = {"text": "".join(texts) if texts else None}
    if any(thinking):
        output["reasoning_content"] = "".join(thinking)
    if response.tool_calls:
        output["tool_calls"] = [attributes({
            "id": call.id, "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.input)},
        }) for call in response.tool_calls]
    if response.citations:
        output["citations"] = [
            {key: val for key, val in {"text": part.text, "title": part.title, "url": part.url}.items() if val is not None}
            for part in response.citations
        ]
    if logprobs:
        output["logprobs"] = attributes({"content": [
            {"token": token.token, "logprob": token.logprob,
             "bytes": list(token.bytes) if token.bytes is not None else None,
             "top_logprobs": [{"token": top.token, "logprob": top.logprob,
                               "bytes": list(top.bytes) if top.bytes is not None else None} for top in token.top]}
            for token in response.logprobs or ()
        ]})
    return output


def usage_dict(response: Response, provider=None):
    u = response.usage
    inputs, outputs = u.input_tokens, u.output_tokens
    # LiteLLM's public prompt/completion counters include cached Anthropic
    # input and Gemini thinking output. lm15's counters are provider-verbatim.
    if provider and "anthropic" in provider and inputs is not None:
        inputs += (u.cache_read_tokens or 0) + (u.cache_write_tokens or 0)
    if provider in {"gemini", "vertex", "vertex-express", "xai"} and outputs is not None:
        outputs += u.reasoning_tokens or 0
    data = {key: val for key, val in {
        "prompt_tokens": inputs, "completion_tokens": outputs,
        "total_tokens": (inputs + outputs if provider and "anthropic" in provider and inputs is not None and outputs is not None
                         else u.total_tokens),
    }.items() if val is not None}
    prompt_details = {key: val for key, val in {
        "cached_tokens": u.cache_read_tokens, "cache_creation_tokens": u.cache_write_tokens,
        "audio_tokens": u.input_audio_tokens,
    }.items() if val is not None}
    completion_details = {key: val for key, val in {
        "reasoning_tokens": u.reasoning_tokens, "audio_tokens": u.output_audio_tokens,
    }.items() if val is not None}
    if data or prompt_details or completion_details:
        data["prompt_tokens_details"] = prompt_details or None
        data["completion_tokens_details"] = completion_details or None
    return data


@dataclass
class CallResult:
    responses: tuple[Response, ...] = ()
    outputs: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    cost: float | None = None
    response_model: str | None = None
    cache_hit: bool = False
    cost_details: dict = field(default_factory=dict)

    @classmethod
    def native(cls, response, *, model_type="chat", logprobs=False, provider=None):
        output = convenience_output(response, logprobs=logprobs)
        outputs = [output] if model_type == "responses" or len(output) > 1 else [output["text"]]
        return cls((response,), outputs, usage_dict(response, provider), response_model=response.model)

    def dump(self):
        return {"_dspy_format": CACHE_FORMAT,
                "responses": [response_to_dict(r, include_provider_data=True) for r in self.responses],
                "outputs": plain(self.outputs), "usage": plain(self.usage), "cost": self.cost,
                "cost_details": plain(self.cost_details), "response_model": self.response_model}

    @classmethod
    def load(cls, record):
        responses = tuple(response_from_dict(r) for r in record["responses"])
        return cls(responses, attributes(record["outputs"]), {}, record["cost"], record["response_model"], True,
                   record.get("cost_details", {}))


def combine(results, *, model_type):
    from dspy.utils.usage_tracker import UsageTracker

    tracker = UsageTracker()
    for result in results:
        tracker.add_usage("combined", result.usage)
    outputs = [out for result in results for out in result.outputs]
    if any(isinstance(out, dict) for out in outputs):
        outputs = [out if isinstance(out, dict) else {"text": out} for out in outputs]
    costs = [r.cost for r in results]
    return CallResult(tuple(r for item in results for r in item.responses), outputs,
                      tracker.get_total_tokens().get("combined", {}),
                      sum(costs) if costs and all(c is not None for c in costs) else None,
                      response_model=results[0].response_model if results else None,
                      cost_details={"candidates": [r.cost_details for r in results]} if any(r.cost_details for r in results) else {})
