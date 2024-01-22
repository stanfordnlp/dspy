---
sidebar_position: 2
---

# Executing Signatures

## Configuring LM

## Executing Signatures

## Inspecting Output

We can actually see how DSPy used our signature to build up the prompt. Let's take the prompt using `inspect_history` method.

## How `Predict` works?

The output of predictor is a `Prediction` class object which is basically same as `Example` class but with some added capabilities most around data creation from llm completions. How does `Predict` module predict though? Here is a step by step breakdown:

1. A call to the predictor will get executed in `__call__` method of `Predict` Module which executes the `forward` method of the class.

2. In `forward` method, DSPy initializes the signature, llm call parameters and few shot examples if any.

3. Generates the output by using generate method which is a DSP primitive and return the output as a `Prediction` object.

4. In generate method, `_generate` method formats into the signature the few shots example and use the lm object we configures to generate the output.

5. In case you are wondering how the prompt is constructed, signature internally takes care of building the prompt structure for which it utilizes the Template class which is another DSP primitive.

Predict gives you a predefined pipeline to execute signature which is nice but you can build much more complicated pipelines with this by creating custom Modules.