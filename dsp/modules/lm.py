from abc import ABC, abstractmethod


class LM(ABC):
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

    def print_green(self, text, end="\n"):
        print("\x1b[32m" + str(text) + "\x1b[0m", end=end)

    def print_red(self, text, end="\n"):
        print("\x1b[31m" + str(text) + "\x1b[0m", end=end)

    def inspect_history(self, n=1):
        provider = self.provider

        last_prompt = None
        printed = []

        for x in reversed(self.history[-100:]):
            prompt = x["prompt"]

            if prompt != last_prompt:
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

        for prompt, choices in reversed(printed):
            print("\n\n\n")
            print(prompt, end="")
            self.print_green(
                choices[0].text if provider == "cohere" else choices[0]["text"], end=""
            )

            if len(choices) > 1:
                self.print_red(f" \t (and {len(choices)-1} other completions)", end="")
            print("\n\n\n")

    @abstractmethod
    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        pass
