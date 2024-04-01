from abc import ABC, abstractmethod


class LM(ABC):
    """Abstract class for language models."""

    def __init__(self, model):
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

        self.history = []

    @abstractmethod
    def basic_request(self, prompt, **kwargs):
        pass

    def request(self, prompt, **kwargs):
        return self.basic_request(prompt, **kwargs)

    def print_green(self, text: str, end: str = "\n"):
        return "\x1b[32m" + str(text) + "\x1b[0m" + end

    def print_red(self, text: str, end: str = "\n"):
        return "\x1b[31m" + str(text) + "\x1b[0m" + end

    def inspect_history(self, n: int = 1, skip: int = 0):
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
                if provider == "clarifai" or provider == "google" or provider == "bedrock-claude":
                    printed.append((prompt, x["response"]))
                elif provider == "anthropic":
                    blocks = [{"text": block.text} for block in x["response"].content if block.type == "text"]
                    printed.append((prompt, blocks))
                elif provider == "cohere":
                    printed.append((prompt, x["response"].text))
                elif provider == "mistral":
                    printed.append((prompt, x['response'].choices))
                else:
                    printed.append((prompt, x["response"]["choices"]))

            last_prompt = prompt

            if len(printed) >= n:
                break

        for idx, (prompt, choices) in enumerate(reversed(printed)):
            printing_value = ""

            # skip the first `skip` prompts
            if (n - idx - 1) < skip:
                continue

            printing_value += "\n\n\n"
            printing_value += prompt

            text = ""
            if provider == "cohere":
                text = choices
            elif provider == "openai" or provider == "ollama":
                text = " " + self._get_choice_text(choices[0]).strip()
            elif provider == "clarifai" or provider == "bedrock-claude":
                text = choices
            elif provider == "google":
                text = choices[0].parts[0].text
            elif provider == "mistral":
                text = choices[0].message.content
            else:
                text = choices[0]["text"]
            printing_value += self.print_green(text, end="")

            if len(choices) > 1:
                printing_value += self.print_red(f" \t (and {len(choices)-1} other completions)", end="")

            printing_value += "\n\n\n"

        print(printing_value)
        return printing_value

    @abstractmethod
    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        pass

    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.kwargs, **kwargs}
        model = kwargs.pop("model")

        return self.__class__(model=model, **kwargs)
