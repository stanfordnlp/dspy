---
sidebar_position: 2
---

# Executing Signatures

Until now we've understood what signatures are and how they build the prompt, however we still have not executed a Signature still. In DSPy, the bare minimum you'll need Signature and `Predict` module to build the most basic pipeline. So let's see how we can use them and what actually happens in them!

## Configuring LM

In order to execute signatures we need `Predict` module which need an LM client to execute the prompt contructed from signatures. So usually in any DSPy script we start by setting up the language model(LM) client. DSPy supports multiple API and local models. In these docs, we'll work with GPT-3.5 (`gpt-3.5-turbo`) unless specified otherwise.

```python
turbo = dspy.OpenAI(model='gpt-3.5-turbo')
dspy.settings.configure(lm=turbo)
```

## Executing Signatures

On it's own the signature is not much, it's the `Predict` module that takes this signature as input and excutes it. Let's see it in action...

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

The `Predict` module uses the LM we configured above to execute the signature that we created. The output i.e. `answer` would be present in the object returned by the `predictor` and can be access via `.` operator.

## Inspecting Output

We can actually see how DSPy used our signature to build up the prompt. Let's take a look at the prompt using `inspect_history` method, once you execute your signature.

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

You can see last `n` prompts executed by LM, however this'll print the prompt executed rather than returning it. If you want to store or use this prompt you can access `history` attribute of the LM object. `history` is the list of dictionaries in which each dictionary is has a `prompt` key storing the prompt feeded to the LLM and a `response` key containing response from the LLM.

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

The output of predictor is a `Prediction` class object which is basically same as `Example` class but with some added capabilities most around data creation from llm completions. How does `Predict` module predict though? Here is a step by step breakdown:

1. A call to the predictor will get executed in `__call__` method of `Predict` Module which executes the `forward` method of the class.

2. In `forward` method, DSPy initializes the signature, llm call parameters and few shot examples if any.

3. Generates the output by using generate method which is a DSP primitive and return the output as a `Prediction` object.

4. In generate method, `_generate` method formats into the signature the few shots example and use the lm object we configures to generate the output.

5. In case you are wondering how the prompt is constructed, signature internally takes care of building the prompt structure for which it utilizes the Template class which is another DSP primitive.

Predict gives you a predefined pipeline to execute signature which is nice but you can build much more complicated pipelines with this by creating custom Modules.