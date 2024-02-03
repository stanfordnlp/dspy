---
sidebar_position: 1
---

# Examples:DSPy :: Tensor:PyTorch

Many frameworks have a "core" data structure. For example, pandas has dataframes and PyTorch utilizes Tensors. In DSPy, we work with **Example**, which are very similar to Python `dict`s but have a few useful utilities.

## Creating an `Example`

When you use DSPy, you will do a lot of evaluation and optimization runs. Your individual datapoints will be of type `Example`. Here's an example to illustrate:

```python
qa_pair = dspy.Example(question="This is a question?", answer="This is an answer.")

print(qa_pair)
print(qa_pair.question)
print(qa_pair.answer)
```
**Output:**
```text
Example({'question': 'This is a question?', 'answer': 'This is an answer.'}) (input_keys=None)
This is a question?
This is an answer.
```
Notice in the code above, we simply initialize the `qa_pair` `Example` object with key-value pairs (e.g. `question` and `answer` acting as arguments). This design approach streamlines data handling within DSPy, allowing for direct access and modification of values using familiar dictionary-like syntax.

```text
object = Example(field1=value1, field2=value2, field3=value3, ...)
```

## Specifying Input Keys

In DSPy, defining inputs keys in `Examples` populates the input to the prompt and are set using the `with_inputs()` method in the `Example` class.

```python
# Single Input
print(qa_pair.with_inputs("question"))

# Multiple Inputs
print(qa_pair.with_inputs("question", "answer"))
```

For a single input, pass the desired key. For multiple inputs, pass multiple keys.

This flexibility allows for customized tailoring of the `Example` object for different DSPy scenarios.

Each `Example` object has an `_input_keys` attribute, updated with each call to `with_inputs()`.

:::caution

If updating input keys through `with_inputs()`, you create a new copy of the current object with an updated `_input_key` attribute. Each call overwrites previous input keys.

**Wrong** ❌
```python
print(qa_pair.with_inputs("question"))  # input_keys: question

# SOME CODE HERE

print(qa_pair.with_inputs("answer"))  # input_keys: answer, question will not be an input_key in this as this'll return a new Example object
```

**Correct** ✅
```python
print(qa_pair.with_inputs("question"))  # input_keys: question

# SOME CODE HERE

print(qa_pair.with_inputs("question", "answer"))  # input_keys: question, answer
```
:::

## Element Access and Updation

Values can be accessed using the `.`(dot) operator. You can access the value of key `name` in defined object `Example(name="John Doe", job="sleep")` through `object.name`. 

To access or exclude certain keys, use `inputs()` and `labels()` methods to return new Example objects containing only input or non-input keys, respectively.

```python
article_summary = dspy.Example(article= "This is an article.", summary= "This is a summary.").with_inputs("article")

input_key_only = article_summary.inputs()
non_input_key_only = article_summary.labels()

print("Example object with Input fields only:", input_key_only)
print("Example object with Non-Input fields only:", non_input_key_only))
```

**Output**
```
Example object with Input fields only: Example({'article': 'This is an article.'}) (input_keys=None)
Example object with Non-Input fields only: Example({'summary': 'This is a summary.'}) (input_keys=None)
```

To exclude keys, use `without()`:

```python
article_summary = dspy.Example(context="This is an article.", question="This is a question?", answer="This is an answer.", rationale= "This is a rationale.").with_inputs("context", "question")

print("Example object without answer & rationale keys:", article_summary.without("answer", "rationale"))
```

**Output**
```
Example object without answer & rationale keys: Example({'context': 'This is an article.', 'question': 'This is a question?'}) (input_keys=None)
```

Updating values is simply assigning a new value using the `.` operator.

```python
article_summary.context = "new context"
```

## Iterating over Example

Iteration in the `Example` class also functions like a dictionary, supporting methods like `keys()`, `values()`, etc: 

```python
for k, v in article_summary.items():
    print(f"{k} = {v}")
```

**Output**

```text
context = This is an article.
question = This is a question?
answer = This is an answer.
rationale = This is a rationale.
```
