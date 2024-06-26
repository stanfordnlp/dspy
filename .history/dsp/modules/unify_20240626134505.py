import os
import requests
import json
from dspy import LM

class Unify(LM):
    def __init__(
        self, 
        endpoint="router@q:1|c:4.65e-03|t:2.08e-05|i:2.07e-03", 
        model_type: Literal["chat", "text"] = "chat",
        system_prompt: Optional[str] = None,
        api_key=None,
    ):
        self.endpoint = endpoint
        self.api_key = api_key or os.getenv("UNIFY_API_KEY")
        self.api_provider: Literal["unify"] = "unify",
        self.api_base = "https://api.unify.ai/v0"
        self.model = endpoint
        super().__init__(model=self.model)
        # self.provider = "unify"
        # openai.api_type = api_provider
        self.system_prompt = system_prompt
        self.model_type = model_type
        self.kwargs = {
            "temperature": 0.0,
            "max_tokens": 200,
            "top_p": 1,
            "top_k": 20,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            "num_ctx": 1024,
            **kwargs,
        }  

        self.kwargs["model"] = model
        self.history: list[dict[str, Any]] = []
        

    def basic_request(self, prompt, **kwargs):
        
        """
        Send request to the Unify AI API.
        This method is required by the LM base class.
        """
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        
        settings_dict = {
            "model": self.model,
            "options": {k: v for k, v in kwargs.items() if k not in ["n", "max_tokens"]},
            "stream": False,
        }
        if self.model_type == "chat":
            settings_dict["messages"] = [{"role": "user", "content": prompt}]
        else:
            settings_dict["prompt"] = prompt
        
        
        
        
        try:
            response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
            return None
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Error parsing API response: {e}")
            return None

    def __call__(self, prompt, **kwargs):
        """Allow the class to be called like a function."""
        return self.basic_request(prompt, **kwargs)

    def generate(self, prompt, max_tokens=100, temperature=0.7, n=1, stop=None):
        """Generate one or more responses to the given prompt."""
        kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
            "stop": stop
        }
        responses = []
        for _ in range(n):
            response = self(prompt, **kwargs)
            if response:
                responses.append(response)
            else:
                print("Failed to generate response")
        return responses