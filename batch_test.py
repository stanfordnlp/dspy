import dspy

class TranslateSignature(dspy.Signature):
    """Translate an English question to French."""
    english_question = dspy.InputField()
    french_question = dspy.OutputField()

class FrenchQASignature(dspy.Signature):
    """Answer a question in French."""
    french_question = dspy.InputField()
    french_answer = dspy.OutputField()

class TranslateAndAnswer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.translator = dspy.Predict(TranslateSignature)
        self.qa = dspy.Predict(FrenchQASignature)

    def forward(self, english_question):
        french_question = self.translator(english_question=english_question).french_question
        french_answer = self.qa(french_question=french_question).french_answer
        return dspy.Prediction(answer=french_answer)

def main():
    # Configure the language model
    # Make sure to set your OPENAI_API_KEY environment variable
    lm = dspy.LM("openai/gpt-3.5-turbo-0125", num_retries=3)
    dspy.settings.configure(lm=lm)

    questions = [
        "What is the capital of France?",
        "Who wrote '1984'?",
        "What is the square root of 16?",
        "What is the boiling point of water in Celsius?",
        "Who discovered penicillin?",
    ]
    examples = [dspy.Example(english_question=q).with_inputs("english_question") for q in questions]
    translate_and_answer = TranslateAndAnswer()

    print("Running multi-step module in batch mode...")
    results = translate_and_answer.batch(examples, batch_mode=True)

    print("\n--- Results ---")
    for i, result in enumerate(results):
        print(f"English Q: {questions[i]}")
        print(f"French A: {result.answer}")
        print("-" * 15)

if __name__ == "__main__":
    main() # uv run batch_test.py
