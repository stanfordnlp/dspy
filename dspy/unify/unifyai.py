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

    def get_credit_balance(self):
        """Get the current credit balance."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.get(f"{self.api_base}/get_credits", headers=headers)
            response.raise_for_status()
            return response.json()["credits"]
        except requests.RequestException as e:
            print(f"Failed to get credit balance: {e}")
            return None

# ... (previous code remains the same)

# Usage example
if __name__ == "__main__":
    # Initialize the UnifyAI instance with a specific model and fallback
    unify_lm = UnifyAI(endpoint="llama-3-8b-chat@fireworks-ai->gpt-3.5-turbo@openai")
    
    # Check credit balance
    credit_balance = unify_lm.get_credit_balance()
    print(f"Current credit balance: {credit_balance}")

    # List available models
    print("Available models:")
    models = unify_lm.list_available_models()
    for model in models:
        print(f"- {model}")

    # Generate a response
    prompt = "Translate 'Hello, world!' to French."
    print(f"\nGenerating response for prompt: '{prompt}'")
    responses = unify_lm.generate(prompt, max_tokens=50, temperature=0.7, n=1)
    
    if responses:
        print("Generated response:")
        for response in responses:
            print(response)
    else:
        print("Failed to generate any responses.")

    # Example with router
    router_lm = UnifyAI(endpoint="router@q:1|c:4.65e-03|t:2.08e-05|i:2.07e-03")
    print("\nUsing router for generation:")
    router_responses = router_lm.generate("What is the capital of France?", max_tokens=50)
    if router_responses:
        print("Router-generated response:")
        for response in router_responses:
            print(response)
    else:
        print("Router failed to generate any responses.")

    # Example with custom endpoint (if applicable)
    # custom_lm = UnifyAI(model_or_router="my-custom-model@custom")
    # custom_response = custom_lm.generate("Custom model prompt", max_tokens=50)
    # print("\nCustom model response:", custom_response)