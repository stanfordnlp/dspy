import datetime
import uuid

from dspy.dsp.utils import settings
from dspy.utils.callback import with_callbacks
from dspy.utils.inspect_history import pretty_print_history

MAX_HISTORY_SIZE = 10_000
GLOBAL_HISTORY = []


class BaseLM:
    """Base class for handling LLM calls.

    Most users can directly use the `dspy.LM` class, which is a subclass of `BaseLM`. Users can also implement their
    own subclasses of `BaseLM` to support custom LLM providers and inject custom logic. To do so, simply override the
    `forward` method and make sure the return format is identical to the
    [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object).

    Example:

    ```python
    from openai import OpenAI

    import dspy


    class MyLM(dspy.BaseLM):
        def forward(self, prompt, messages=None, **kwargs):
            client = OpenAI()
            return client.chat.completions.create(
                model=self.model,
                messages=messages or [{"role": "user", "content": prompt}],
                **self.kwargs,
            )


    lm = MyLM(model="gpt-4o-mini")
    dspy.configure(lm=lm)
    print(dspy.Predict("q->a")(q="Why did the chicken cross the kitchen?"))
    ```
    """

    def __init__(self, model, model_type="chat", temperature=0.0, max_tokens=1000, cache=True, **kwargs):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []

    def _process_lm_response(self, response, prompt, messages, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}

        if self.model_type == "responses":
            outputs = self._process_response(response)
        else:
            outputs = self._process_completion(response, merged_kwargs)

        if settings.disable_history:
            return outputs

        # Logging, with removed api key & where `cost` is None on cache hit.
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        entry = {
            "prompt": prompt,
            "messages": messages,
            "kwargs": kwargs,
            "response": response,
            "outputs": outputs,
            "usage": dict(response.usage),
            "cost": getattr(response, "_hidden_params", {}).get("response_cost"),
            "timestamp": datetime.datetime.now().isoformat(),
            "uuid": str(uuid.uuid4()),
            "model": self.model,
            "response_model": response.model,
            "model_type": self.model_type,
        }

        self.update_history(entry)

        return outputs

    @with_callbacks
    def __call__(self, prompt=None, messages=None, **kwargs):
        response = self.forward(prompt=prompt, messages=messages, **kwargs)
        outputs = self._process_lm_response(response, prompt, messages, **kwargs)

        return outputs

    @with_callbacks
    async def acall(self, prompt=None, messages=None, **kwargs):
        response = await self.aforward(prompt=prompt, messages=messages, **kwargs)
        outputs = self._process_lm_response(response, prompt, messages, **kwargs)
        return outputs

    def forward(self, prompt=None, messages=None, **kwargs):
        """Forward pass for the language model.

        Subclasses must implement this method, and the response should be identical to
        [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    async def aforward(self, prompt=None, messages=None, **kwargs):
        """Async forward pass for the language model.

        Subclasses that support async should implement this method, and the response should be identical to
        [OpenAI response format](https://platform.openai.com/docs/api-reference/responses/object).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def copy(self, **kwargs):
        """Returns a copy of the language model with possibly updated parameters.

        Any provided keyword arguments update the corresponding attributes or LM kwargs of
        the copy. For example, ``lm.copy(rollout_id=1, temperature=1.0)`` returns an LM whose
        requests use a different rollout ID at non-zero temperature to bypass cache collisions.
        """

        import copy

        new_instance = copy.deepcopy(self)
        new_instance.history = []

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(new_instance, key, value)
            if (key in self.kwargs) or (not hasattr(self, key)):
                if value is None:
                    new_instance.kwargs.pop(key, None)
                else:
                    new_instance.kwargs[key] = value
        if hasattr(new_instance, "_warned_zero_temp_rollout"):
            new_instance._warned_zero_temp_rollout = False

        return new_instance

    def inspect_history(self, n: int = 1):
        return pretty_print_history(self.history, n)

    def update_history(self, entry):
        if settings.disable_history:
            return

        # Global LM history
        if len(GLOBAL_HISTORY) >= MAX_HISTORY_SIZE:
            GLOBAL_HISTORY.pop(0)

        GLOBAL_HISTORY.append(entry)

        if settings.max_history_size == 0:
            return

        # dspy.LM.history
        if len(self.history) >= settings.max_history_size:
            self.history.pop(0)

        self.history.append(entry)

        # Per-module history
        caller_modules = settings.caller_modules or []
        for module in caller_modules:
            if len(module.history) >= settings.max_history_size:
                module.history.pop(0)
            module.history.append(entry)

    def _process_completion(self, response, merged_kwargs):
        """Process the response of OpenAI chat completion API and extract outputs.

        Args:
            response: The OpenAI chat completion response
                https://platform.openai.com/docs/api-reference/chat/object
            merged_kwargs: Merged kwargs from self.kwargs and method kwargs

        Returns:
            List of processed outputs
        """
        outputs = []
        for c in response.choices:
            output = {}
            output["text"] = c.message.content if hasattr(c, "message") else c["text"]
            if merged_kwargs.get("logprobs"):
                output["logprobs"] = c.logprobs if hasattr(c, "logprobs") else c["logprobs"]
            if hasattr(c, "message") and getattr(c.message, "tool_calls", None):
                output["tool_calls"] = c.message.tool_calls

            # Extract citations from LiteLLM response if available
            citations = self._extract_citations_from_response(c)
            if citations:
                output["citations"] = citations

            outputs.append(output)

        if all(len(output) == 1 for output in outputs):
            # Return a list if every output only has "text" key
            outputs = [output["text"] for output in outputs]

        return outputs

    def _extract_citations_from_response(self, choice):
        """Extract citations from LiteLLM response if available.
        Reference: https://docs.litellm.ai/docs/providers/anthropic#beta-citations-api

        Args:
            choice: The choice object from response.choices

        Returns:
            A list of citation dictionaries or None if no citations found
        """
        try:
            # Check for citations in LiteLLM provider_specific_fields
            citations_data = choice.message.provider_specific_fields.get("citations")
            if isinstance(citations_data, list):
                return [citation for citations in citations_data for citation in citations]
        except Exception:
            return None

    def _process_response(self, response):
        """Process the response of OpenAI Response API and extract outputs.

        Args:
            response: OpenAI Response API response
                https://platform.openai.com/docs/api-reference/responses/object

        Returns:
            List of processed outputs, which is always of size 1 because the Response API only supports one output.
        """
        text_outputs = []
        tool_calls = []
        reasoning_contents = []

        for output_item in response.output:
            output_item_type = output_item.type
            if output_item_type == "message":
                for content_item in output_item.content:
                    text_outputs.append(content_item.text)
            elif output_item_type == "function_call":
                tool_calls.append(output_item.model_dump())
            elif output_item_type == "reasoning":
                if getattr(output_item, "content", None) and len(output_item.content) > 0:
                    for content_item in output_item.content:
                        reasoning_contents.append(content_item.text)
                elif getattr(output_item, "summary", None) and len(output_item.summary) > 0:
                    for summary_item in output_item.summary:
                        reasoning_contents.append(summary_item.text)

        result = {}
        if len(text_outputs) > 0:
            result["text"] = "".join(text_outputs)
        if len(tool_calls) > 0:
            result["tool_calls"] = tool_calls
        if len(reasoning_contents) > 0:
            result["reasoning_content"] = "".join(reasoning_contents)
        # All `response.output` items map to one answer, so we return a list of size 1.
        return [result]


def inspect_history(n: int = 1):
    """The global history shared across all LMs."""
    return pretty_print_history(GLOBAL_HISTORY, n)
