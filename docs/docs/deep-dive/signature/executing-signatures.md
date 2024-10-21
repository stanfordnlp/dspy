---
sidebar_position: 2
---

# Executing Signatures

So far we've understood what signatures are and how we can use them to craft our prompt, but now let's take a look at how to execute them.

## Configuring LM

To execute signatures, we require DSPy modules which are themselves dependent on a client connection to a language model (LM) client. DSPy supports LM APIs and local model hosting. Within this example, we will make use of the OpenAI client and configure the GPT-3.5 (`gpt-3.5-turbo`) model.

```python
turbo = dspy.OpenAI(model='gpt-3.5-turbo')
dspy.settings.configure(lm=turbo)
```

## Executing Signatures

Let's make use of the simplest module in DSPy - the `Predict` module that takes this signature as input to construct the prompt sent to the LM and generates a response for it. 

```python
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# Define the predictor.
predictor = dspy.Predict(BasicQA)

# Call the predictor on a particular input.
pred = predictor(question=devset[0].question)

# Print the input and the prediction.
print(f"Question: {devset[0].question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Actual Answer: {devset[0].answer}")
```
**Output:**
```text
Question: Are both Cangzhou and Qionghai in the Hebei province of China?
Predicted Answer: No.
Actual Answer: no
```

The `Predict` module generates a response via the LM we configured above and executes the prompt crafted by the signature. This returns the output i.e. `answer` which is present in the object returned by the `predictor` and can be accessed via `.` operator.

## Inspecting Output

Let's dive deeper into DSPy uses our signature to build up the prompt, which we can do through the `inspect_history` method on the configured LM following the program's execution. This method returns the last `n` prompts executed by LM.

```python
turbo.inspect_history(n=1)
```
**Output:**
```text
Answer questions with short factoid answers.

---

Follow the following format.

Question: ${question}
Answer: often between 1 and 5 words

---

Question: Are both Cangzhou and Qionghai in the Hebei province of China?
Answer: No.
```

Additionally, if you want to store or use this prompt, you can access the `history` attribute of the LM object, which stores a list of dictionaries containing respective  `prompt:response` entries for each LM generation.

```python
turbo.history[0]
```
**Output:**
```text
{'prompt': "Answer questions with short factoid answers.\n\n---\n\nFollow the following format.\n\nQuestion: ${question}\nQuestion's Answer: often between 1 and 5 words\n\n---\n\nQuestion: Are both Cangzhou and Qionghai in the Hebei province of China?\nQuestion's Answer:",
 'response': <OpenAIObject chat.completion id=chatcmpl-8kCPsxikpVpmSaxdGLUIqubFZS05p at 0x7c3ba41fa840> JSON: {
   "id": "chatcmpl-8kCPsxikpVpmSaxdGLUIqubFZS05p",
   "object": "chat.completion",
   "created": 1706021508,
   "model": "gpt-3.5-turbo-0613",
   "choices": [
     {
       "index": 0,
       "message": {
         "role": "assistant",
         "content": "No."
       },
       "logprobs": null,
       "finish_reason": "stop"
     }
   ],
   "usage": {
     "prompt_tokens": 64,
     "completion_tokens": 2,
     "total_tokens": 66
   },
   "system_fingerprint": null
 },
 'kwargs': {'stringify_request': '{"temperature": 0.0, "max_tokens": 150, "top_p": 1, "frequency_penalty": 0, "presence_penalty": 0, "n": 1, "model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "Answer questions with short factoid answers.\\n\\n---\\n\\nFollow the following format.\\n\\nQuestion: ${question}\\nQuestion\'s Answer: often between 1 and 5 words\\n\\n---\\n\\nQuestion: Are both Cangzhou and Qionghai in the Hebei province of China?\\nQuestion\'s Answer:"}]}'},
 'raw_kwargs': {}}
```

## How `Predict` works?

The output of predictor is a `Prediction` class object which mirrors the `Example` class with additional functionalities for LM completion interactivity. 

How does `Predict` module actually 'predict' though? Here is a step-by-step breakdown:

1. A call to the predictor will get executed in `__call__` method of `Predict` Module which executes the `forward` method of the class.

2. In `forward` method, DSPy initializes the signature, LM call parameters and few-shot examples, if any.

3. The `_generate` method formats the few shots example to mirror the signature and uses the LM object we configured to generate the output as a `Prediction` object. 

In case you are wondering how the prompt is constructed, the DSPy Signature framework internally handles the prompt structure, utilizing the DSP Template primitive to craft the prompt. 

Predict gives you a predefined pipeline to execute signature which is nice but you can build much more complicated pipelines with this by creating custom Modules.
