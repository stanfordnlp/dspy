import os
import requests
import json
from dspy import LM

class UnifyAI(LM):
    def __init__(self, model_or_router="auto", api_key=None):
        # Initialize API key and base URL
        self.api_key = api_key or os.getenv("UNIFY_API_KEY")
        self.api_base = "https://api.unify.ai/v0"
        
        # Set the model or router
        self.set_model_or_router(model_or_router)
        
        # Initialize the base LM class with the selected model
        super().__init__(model=self.model)

    def set_model_or_router(self, model_or_router):
        """Set the model or router based on the input."""
        if model_or_router == "auto":
            self.model = "router@auto"
        elif model_or_router.startswith("router@"):
            self.model = model_or_router
        else:
            self.model = model_or_router

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
            # Send POST request to the API
            response = requests.post(f"{self.api_base}/chat/completions", headers=headers, json=data)
            response.raise_for_status()
            
            # Extract and return the generated content
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

    def list_available_models(self):
        """Retrieve a list of available models from the API."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.get(f"{self.api_base}/models", headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Failed to list models: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
            return []

# Usage example
if __name__ == "__main__":
    # Initialize the UnifyAI instance
    unify_lm = UnifyAI()
    
    # List available models
    print("Available models:")
    print(unify_lm.list_available_models())
    
    print("\nGenerating response:")
    response = unify_lm.generate("Translate 'Hello, world!' to French.", max_tokens=50)
    print("Generated response:", response)