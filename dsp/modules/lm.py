from abc import ABC, abstractmethod
import base64
import requests
from typing import Union, List, Dict

class LM(ABC):
    """Abstract class for language models, now with enhanced GPT-4 Vision support."""

    def __init__(self, model, api_key):
        self.kwargs = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": 150,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
        }
        self.provider = "openai"
        self.api_key = api_key
        self.history = []

    def encode_image_to_base64(self, image_path: str) -> str:
        """Encodes an image to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def prepare_prompt(self, prompt: Union[str, List[Dict[str, Union[str, Dict]]]]):
        """Enhanced to explicitly handle both text and base64/image URL prompts for GPT-4 Vision."""
        prepared_prompts = []
        if isinstance(prompt, str):
            prepared_prompts.append({"role": "user", "content": prompt})
        else:
            for item in prompt:
                if item['type'] == 'text':
                    prepared_prompts.append({"role": "user", "content": item['content']})
                elif item['type'] == 'image':
                    # Assuming 'content' contains the path to the image for base64 encoding
                    base64_image = self.encode_image_to_base64(item['content'])
                    prepared_prompts.append({
                        "role": "user",
                        "content": {
                            "image": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                elif item['type'] == 'image_url':
                    prepared_prompts.append({
                        "role": "user",
                        "content": {
                            "image_url": item['content']
                        }
                    })
        return prepared_prompts

    def request(self, prompts: List[Dict[str, Union[str, Dict]]]):
        """Send request to OpenAI API, including handling for both text and image inputs."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.kwargs["model"],
            "messages": self.prepare_prompt(prompts),
            **self.kwargs
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
        if response.status_code == 200:
            self.history.append(response.json())
            return response.json()
        else:
            raise Exception(f"Request failed: {response.text}")

    def inspect_history(self, n: int = 1, skip: int = 0):
        """Prints the last n prompts and their completions, including handling for image inputs."""
        provider: str = self.provider

        last_prompt = None
        printed = []
        n = n + skip

        for x in reversed(self.history[-100:]):
            prompt = x["prompt"]

            if prompt != last_prompt:
                if provider == "clarifai" or provider == "google" or provider == "claude":
                    printed.append((prompt, x['response']))
                else:
                    printed.append((prompt, x["response"].generations if provider == "cohere" else x["response"]["choices"]))

            last_prompt = prompt

            if len(printed) >= n:
                break

        for idx, (prompt, choices) in enumerate(reversed(printed)):
            if (n - idx - 1) < skip:
                continue

            print("\n\n\n")
            print(f"Prompt: {prompt}", end="\n\n")
            for choice in choices:
                text = choice if isinstance(choice, str) else "Image response or complex structure"  # Placeholder for complex handling
                self.print_green(text, end="\n\n")

    @abstractmethod
    def basic_request(self, prompt: Union[str, Dict, List[Dict[str, Union[str, Dict]]]], **kwargs):
        """To be implemented by subclasses for making requests."""
        pass

    @abstractmethod
    def __call__(self, prompts: Union[str, List[Dict[str, Union[str, Dict]]]], **kwargs):
        """Abstract method to call the model with enhanced prompts."""
        pass

    def copy(self, **kwargs):
        """Returns a copy of the language model with the same parameters."""
        new_kwargs = {**self.kwargs, **kwargs}
        model = new_kwargs.pop('model')
        return self.__class__(model=model, api_key=self.api_key, **new_kwargs)
