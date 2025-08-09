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

        outputs = []
        for c in response.choices:
            output = {}
            output["text"] = c.message.content if hasattr(c, "message") else c["text"]
            if merged_kwargs.get("logprobs"):
                output["logprobs"] = c.logprobs if hasattr(c, "logprobs") else c["logprobs"]
            if hasattr(c, "message") and getattr(c.message, "tool_calls", None):
                output["tool_calls"] = c.message.tool_calls
            outputs.append(output)

        if all(len(output) == 1 for output in outputs):
            # Return a list if every output only has "text" key
            outputs = [output["text"] for output in outputs]

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
        """Returns a copy of the language model with possibly updated parameters."""

        import copy

        new_instance = copy.deepcopy(self)
        new_instance.history = []

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(new_instance, key, value)
            if (key in self.kwargs) or (not hasattr(self, key)):
                new_instance.kwargs[key] = value

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


def inspect_history(n: int = 1):
    """The global history shared across all LMs."""
    return pretty_print_history(GLOBAL_HISTORY, n)
