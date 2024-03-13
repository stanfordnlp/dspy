from vertexai.generative_models import GenerativeModel

model = GenerativeModel("gemini-1.0-pro")

response = model.generate_content("Hello")
print(response)
