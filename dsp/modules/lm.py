from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass#(frozen=True) technically should be frozen
class LM(ABC):
    """Abstract class for language models."""
    kwargs: dict[str, Any] = field(init=False)
    provider: str = field(init=False)
    history: list[dict[str, Any]] = field(init=False)

    def __init__(self, model):
        object.__setattr__(self, 'kwargs', {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
        })
        object.__setattr__(self, 'provider', "default")
        object.__setattr__(self, 'history', [])

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

    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        kwargs = {**self.kwargs, **kwargs}
        model = kwargs.pop("model")

        return self.__class__(model=model, **kwargs)
    
class FinetuneableLM(LM, ABC):
    def verify_training_arguments(self, dataset: dict[str, Any], valset: Optional[dict[str, Any]], training_arguments: dict[str, Any]) -> bool:
        return True

    @abstractmethod
    def format_data_for_vanilla_finetuning(self, data: list[dict[str, str]]) -> list[dict[str, str]]:
        pass

    @abstractmethod
    def _load_data(self, train_path: str, val_path: Optional[str]) -> str:
        pass

    @abstractmethod
    def start_training(self, **kwargs) -> None:
        pass

    @abstractmethod
    def stop_training(self) -> None:
        pass

    @abstractmethod
    def check_training_status(self) -> bool:
        pass

    @abstractmethod
    def retrieve_trained_model_client(self) -> LM:
        pass