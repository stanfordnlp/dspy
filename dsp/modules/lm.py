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
        print("\x1b[32m" + str(text) + "\x1b[0m", end=end)

    def print_red(self, text: str, end: str = "\n"):
        print("\x1b[31m" + str(text) + "\x1b[0m", end=end)

    def inspect_history(self, n: int = 1, skip: int = 0):
        """Prints the last n prompts and their completions.
        TODO: print the valid choice that contains filled output field instead of the first
        """
        provider: str = self.provider

        last_prompt = None
        printed = []
        n = n + skip

        for x in reversed(self.history[-100:]):
            prompt = x["prompt"]

            if prompt != last_prompt:

                if provider == "clarifai" or provider == "google":
                    printed.append(
                        (
                            prompt,
                            x['response']
                        )
                    )
                else:
                    printed.append(
                        (
                            prompt,
                            x["response"].generations
                            if provider == "cohere"
                            else x["response"]["choices"],
                        )
                    )

            last_prompt = prompt

            if len(printed) >= n:
                break

        for idx, (prompt, choices) in enumerate(reversed(printed)):
            # skip the first `skip` prompts
            if (n - idx - 1) < skip:
                continue

            print("\n\n\n")
            print(prompt, end="")
            text = ""
            if provider == "cohere":
                text = choices[0].text
            elif provider == "openai" or provider == "ollama":
                text = ' ' + self._get_choice_text(choices[0]).strip()
            elif provider == "clarifai":
                text=choices
            elif provider == "google":
                text = choices[0].parts[0].text
            else:
                text = choices[0]["text"]
            self.print_green(text, end="")

            if len(choices) > 1:
                self.print_red(f" \t (and {len(choices)-1} other completions)", end="")
            print("\n\n\n")

    @abstractmethod
    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        pass

    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.kwargs, **kwargs}
        model = kwargs.pop('model')

        return self.__class__(model, **kwargs)
