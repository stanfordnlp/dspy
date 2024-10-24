from abc import ABC, abstractmethod


class LM(ABC):
    """Abstract class for language models."""

    def __init__(self, model, tracker=None):
        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
        }
        self.provider = "default"
        self.tracker = tracker

        self.history = []

    @abstractmethod
    def basic_request(self, prompt, **kwargs):
        pass

    def request(self, prompt, **kwargs):
        return self.basic_request(prompt, **kwargs)

    def print_green(self, text: str, end: str = "\n"):
        import dspy

        if dspy.settings.experimental:
            return "\n\n" + "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end
        else:
            return "\x1b[32m" + str(text) + "\x1b[0m" + end

    def print_red(self, text: str, end: str = "\n"):
        return "\x1b[31m" + str(text) + "\x1b[0m" + end

    def inspect_history(self, n: int = 1, skip: int = 0, color_format: bool = True):
        """Prints the last n prompts and their completions.

        TODO: print the valid choice that contains filled output field instead of the first.
        """
        provider: str = self.provider

        last_prompt = None
        printed = []
        n = n + skip

        for x in reversed(self.history[-100:]):
            prompt = x["prompt"]

            if prompt != last_prompt:
                if provider in (
                    "clarifai",
                    "cloudflare",
                    "google",
                    "groq",
                    "Bedrock",
                    "Sagemaker",
                    "premai",
                    "tensorrt_llm",
                ):
                    printed.append((prompt, x["response"]))
                elif provider == "anthropic":
                    blocks = [
                        {"text": block.text}
                        for block in x["response"].content
                        if block.type == "text"
                    ]
                    printed.append((prompt, blocks))
                elif provider == "cohere":
                    printed.append((prompt, x["response"].text))
                elif provider == "mistral":
                    printed.append((prompt, x["response"].choices))
                elif provider == "ibm":
                    printed.append((prompt, x))
                elif provider == "you.com":
                    printed.append((prompt, x["response"]["answer"]))
                else:
                    printed.append((prompt, x["response"]["choices"]))

            last_prompt = prompt

            if len(printed) >= n:
                break

        printing_value = ""
        for idx, (prompt, choices) in enumerate(reversed(printed)):
            # skip the first `skip` prompts
            if (n - idx - 1) < skip:
                continue
            printing_value += "\n\n\n"
            printing_value += prompt

            text = ""
            if provider in (
                "cohere",
                "Bedrock",
                "Sagemaker",
                "clarifai",
                "claude",
                "ibm",
                "premai",
                "you.com",
                "tensorrt_llm",
            ):
                text = choices
            elif provider == "openai" or provider == "ollama" or provider == "llama":
                text = " " + self._get_choice_text(choices[0]).strip()
            elif provider == "groq":
                text = " " + choices
            elif provider == "google":
                text = choices[0].parts[0].text
            elif provider == "mistral":
                text = choices[0].message.content
            elif provider == "cloudflare":
                text = choices[0]
            else:
                text = choices[0]["text"]
            printing_value += self.print_green(text, end="") if color_format else text

            if len(choices) > 1 and isinstance(choices, list):
                choices_text = f" \t (and {len(choices)-1} other completions)"
                printing_value += self.print_red(
                   choices_text, end="",
                ) if color_format else choices_text

            printing_value += "\n\n\n"

        print(printing_value)
        return printing_value

    @abstractmethod
    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        pass

    def tracker_call(self, tracker, prompt=None, output=None, name=None, **kwargs):
        from dsp.trackers.base import BaseTracker
        assert issubclass(tracker.__class__, BaseTracker), "tracker must be a subclass of BaseTracker"
        assert self.history, "tracker.call() requires a previous request"

        last_req = self.history[-1]
        if not prompt:
            prompt = last_req.get('prompt', None)
        if not output:
            output = last_req.get('response', None)
        kwargs = {**self.kwargs, **kwargs}
        name = name if name else self.__class__.__name__
        tracker.call(i=prompt, o=output, name=name, **kwargs)

    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.kwargs, **kwargs}
        model = kwargs.pop("model")

        return self.__class__(model=model, **kwargs)
