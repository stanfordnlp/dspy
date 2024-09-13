import dspy

lm = dspy.OpenAI(model="o1-preview", api_key="sk-proj-X", model_type="chat")

dspy.settings.configure(lm=lm)

x = dspy.Predict("question -> answer")

print("Answer:", x(question="What is 1 + 1?").answer)
# Answer: 2
