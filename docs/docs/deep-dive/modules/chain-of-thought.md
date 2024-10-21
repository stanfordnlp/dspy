# dspy.ChainOfThought

### Constructor

The constructor initializes the `ChainOfThought` class and sets up its attributes. It inherits from the `Predict` class and adds specific functionality for chain of thought processing. 

Internally, the class initializes the `activated` attribute to indicate if chain of thought processing has been selected. It extends the `signature` to include additional reasoning steps and an updated `rationale_type` when chain of thought processing is activated.

```python
class ChainOfThought(Predict):
    def __init__(self, signature, rationale_type=None, activated=True, **config):
        super().__init__(signature, **config)

        self.activated = activated

        signature = ensure_signature(self.signature)
        *_keys, last_key = signature.output_fields.keys()

        rationale_type = rationale_type or dspy.OutputField(
            prefix="Reasoning: Let's think step by step in order to",
            desc="${produce the " + last_key + "}. We ...",
        )

        self.extended_signature = signature.prepend("rationale", rationale_type, type_=str)
```

**Parameters:**
- `signature` (_Any_): Signature of predictive model.
- `rationale_type` (_dspy.OutputField_, _optional_): Rationale type for reasoning steps. Defaults to `None`.
- `activated` (_bool_, _optional_): Flag for activated chain of thought processing. Defaults to `True`.
- `**config` (_dict_): Additional configuration parameters for model.

### Method

#### `forward(self, **kwargs)`

This method extends the parent `Predict` class' forward pass while updating the signature when chain of thought reasoning is activated or if the language model is a GPT3 model.

**Parameters:**
- `**kwargs`: Keyword arguments required for prediction.

**Returns:**
- The result of the `forward` method.

### Examples

```python
#Define a simple signature for basic question answering
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

#Pass signature to ChainOfThought module
generate_answer = dspy.ChainOfThought(BasicQA)

# Call the predictor on a particular input.
question='What is the color of the sky?'
pred = generate_answer(question=question)

print(f"Question: {question}")
print(f"Predicted Answer: {pred.answer}")
```

The following example shows how to specify your custom rationale. Here `answer` corresponds to the last key to produce, it may be different in your case. 

```python
#define a custom rationale
rationale_type = dspy.OutputField(
            prefix="Reasoning: Let's think step by step in order to",
            desc="${produce the answer}. We ...",
        )
#Pass signature to ChainOfThought module
generate_answer = dspy.ChainOfThought(BasicQA, rationale_type=rationale_type)
```