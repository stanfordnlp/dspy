import dspy

lm = dspy.LM("gpt-4o-mini")
dspy.settings.configure(lm=lm)

predictor = dspy.Predict("query: str, image: Image -> response: str")
predictor(query="What is this dog?", image=dspy.Image.from_url("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"))

dspy.inspect_history()