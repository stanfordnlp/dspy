from dsp.modules.cohere import Cohere

api_key = "RShnIWkqv26t01ZBVRkVj4TbGh4qyyy1N6ehCYuq"
model = "command-r"

# Initialize the Cohere class
cohere = Cohere(model=model, api_key=api_key)

# Test the __call__ method
prompt = "What is the capital of France?"
response = cohere(prompt)

print("Prompt:")
print(prompt)
print("\nResponse:")
print(response)

# Test the request method directly
prompt = "What is the largest planet in our solar system?"
response = cohere.request(prompt)

print("\nPrompt:")
print(prompt)
print("\nResponse:")
print(response.text)