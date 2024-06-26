import os
import requests
import json
from dspy import LM

class UnifyAI(LM):
    def __init__(self, endpoint="router@q:1|c:4.65e-03|t:2.08e-05|i:2.07e-03", api_key=None):
        self.api_key = api_key or os.getenv("UNIFY_API_KEY")
        self.api_base = "https://api.unify.ai/v0"
        self.set_model_or_router(endpoint)
        super().__init__(model=self.model)

    def set_model_or_router(self, endpoint):
        """Set the model or router based on the input."""
        self.model = endpoint

    def basic_request(self, prompt, **kwargs):
        """
        Send a basic request to the Unify AI API.
        This method is required by the LM base class.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs
        }
        
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