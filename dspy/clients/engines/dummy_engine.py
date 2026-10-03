"""Canonical engine used by DummyLM; no provider SDK objects are needed."""

from dspy.clients.engines.base import validate_request
from dspy.lm15 import Message, Response, TextPart, ThinkingPart, Usage, response_to_events


class DummyEngine:
    def __init__(self, owner):
        self.owner = owner

    def complete(self, request):
        validate_request(request)
        messages = []
        if request.system is not None:
            system = request.system if isinstance(request.system, str) else "".join(p.text for p in request.system)
            messages.append({"role": "system", "content": system})
        messages.extend({"role": m.role, "content": m.text or ""} for m in request.messages)
        return self._complete_messages(messages)

    def complete_legacy(self, lm, request, *, prompt=None, messages=None, **context):
        """Scripted answers do not require provider-compatible input blocks.

        Keep the original messages (including filenames and opaque blocks).
        The dummy generates canonical responses, but does not send inputs to a
        provider or pretend those inputs round-trip through a narrower schema.
        """
        from dspy.clients.call_result import CallResult, combine

        if messages is None and isinstance(prompt, (list, tuple)) and all(
            isinstance(message, dict) and "role" in message for message in prompt
        ):
            messages = prompt
        else:
            messages = request["messages"]
        count = request.get("n") or 1
        results = [CallResult.native(self._complete_messages(messages)) for _ in range(count)]
        return results[0] if count == 1 else combine(results, model_type="chat")

    def _complete_messages(self, messages):
        owner = self.owner
        if owner.follow_examples:
            output = owner._use_example(messages)
        elif isinstance(owner.answers, dict):
            output = next((owner._format_answer_fields(v) for k, v in owner.answers.items()
                           if k in messages[-1]["content"]), "No more responses")
        else:
            output = owner._format_answer_fields(next(owner.answers, {"answer": "No more responses"}))
        parts = [TextPart(output or "")]
        if owner.reasoning:
            parts.insert(0, ThinkingPart("Some reasoning"))
        return Response(id=None, model="dummy", message=Message.assistant(parts), finish_reason="stop",
                        usage=Usage(input_tokens=0, output_tokens=0, total_tokens=0))

    def stream(self, request):
        return response_to_events(self.complete(request))

    def close(self):
        pass


class AsyncDummyEngine:
    def __init__(self, sync):
        self.sync = sync

    async def complete(self, request):
        return self.sync.complete(request)

    async def complete_legacy(self, lm, request, **context):
        return self.sync.complete_legacy(lm, request, **context)

    async def stream(self, request):
        for event in self.sync.stream(request):
            yield event

    async def aclose(self):
        pass
