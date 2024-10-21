---
sidebar_position: 2
---

# ChainOfThoughtWithHint

This class builds upon the `ChainOfThought` class by introducing an additional input field to provide hints for reasoning. The inclusion of a hint allows for a more directed problem-solving process, which can be especially useful in complex scenarios.


`ChainOfThoughtWithHint` is instantiated with a user-defined DSPy Signature, and the inclusion of a `hint` argument that takes in a string-form phrase to provide a hint within the prompt template.

Let's take a look at an example:

```python
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

#Pass signature to ChainOfThought module
generate_answer = dspy.ChainOfThoughtWithHint(BasicQA)

# Call the predictor on a particular input alongside a hint.
question='What is the color of the sky?'
hint = "It's what you often see during a sunny day."
pred = generate_answer(question=question, hint=hint)

print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")
```
