import os
import requests
from dspy.models import LM

class UnifyAI(LM):
    def __init__(self, model_or_router="auto", api_key=None):
        super().__init__()
        self.api_key = api_key or os.getenv("UNIFY_API_KEY")
        self.api_base = "https://api.unify.ai/v0"
        self.set_model_or_router(model_or_router)

    def set_model_or_router(self, model_or_router):
        if model_or_router == "auto":
            self.model = "router@auto"
        elif model_or_router.startswith("router@"):
            self.model = model_or_router
        else:
            self.model = model_or_router

    def __call__(self, prompt, **kwargs):
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
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return None
        
        return response.json()["choices"][0]["message"]["content"]

    def generate(self, prompt, max_tokens=100, temperature=0.7, n=1, stop=None):
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
        return responses

    def list_available_models(self):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            response = requests.get(f"{self.api_base}/models", headers=headers)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to list models: {e}")
            return []
        
        return response.json()

# Usage example
if __name__ == "__main__":
    unify_lm = UnifyAI()
    print(unify_lm.list_available_models())
