---
sidebar_position: 1
---

# Examples:DSPy :: Tensor:PyTorch

If you are familiar with frameworks like NumPy, Pandas or PyTorch you might notice a similarity between them. The similarity can be suble but aside from the more sophisticated answer, simple answer is that they all have there own data structure that acts as the core data currency in that framework. Numpy have arrays, Pandas have dataframes, PyTorch has Tensor and DSPy has **Example**.

## Creating an Example Object

Just how tensors are data currency in PyTorch, Examples are data currency in DSPy. The inputs to your Modules is an input of type `Example``, a class defined in DSPy which in many ways works same as that of a Python Dictionary. Let's take an example below to understand them:-

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

Take a look at the code above, we initialize the Example object `qa_pair` by passing the key-value pairs as arguments. Here the keys are `question`, `answer` and values are `This is a question?`, `This is an answer.`. Poor choice of values aside, you might be wondering why are we calling them key-value pairs and not just arguments?

That's because `Example` is implemented in a way which is structurally similar to dictionaries, we'll see how that is so. In the `Inside Example` blog we'll be seeing in more detail how `Example` class does this and adds a few features to it. Accessing values of the key using `.` operator being one of them, it's kinda like pydantic. So in short if you want to initialize an `Example` object just add key-value pair as argument:

```text
object = Example(field1=value1, field2=value2, field3=value3, ...)
```

## Specifying Input Keys

In DSPy we need to define the inputs keys in examples. These input keys will fill up the input to the the prompt itself and can be defined via `with_inputs()` method in DSPy Example class.

```python
# Single Input
print(qa_pair.with_inputs("question"))

# Multiple Inputs
print(qa_pair.with_inputs("question", "answer"))
```

As you can see, for a single input we simply pass the desired key as an argument. When dealing with multiple inputs, we can pass multiple keys as arguments. This flexibility ensures that the Example object can be tailored to various scenarios and requirements within the DSPy framework.

Each `Example` object has an attribute `_input_keys` which is a set that get updated when you call it each time. We'll see what happens in detail inside `with_inputs()` in further sections, however to feed our curiosity I thought I should explain it a bit.

:::caution

In an unusual case, if you are updating input keys using the `with_inputs()` method, you are creating a new creating a copy of the current object with updated `_input_key` attribute so each time you call it you'll overwrite on input_keys you added previously. For Example:

**Wrong Way** ❌
```python
print(qa_pair.with_inputs("question"))  # input_keys: question

# SOME CODE HERE

print(qa_pair.with_inputs("answer"))  # input_keys: answer, question will not be an input_key in this as this'll return a new Example object
```

**Correct Way** ✅
```python
print(qa_pair.with_inputs("question"))  # input_keys: question

# SOME CODE HERE

print(qa_pair.with_inputs("question", "answer"))  # input_keys: question, answer
```
:::

## Element Access and Updation

As we saw in examples above values for the keys can be accessed via `.`(dot) operator. So if we have an example object `Example(name="John Doe", job="sleep", )` to access value of key `name` you just call `object.name`. It's quite simple right? What if you need to get all the input keys or maybe none input keys? Luckily, DSPy `Example` class provide us two methods `inputs()` and `labels()` that'll return new Example objects with inputs keys only and non-input keys only respectively.

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

Note that these methods will return a new `Example` object all together so that's something that you need to keep in mind while working with them. Now what if you want an `Example` without a few particular keys? You can use the `without()` method and pass the keys you wish to exclude in it.

```python
article_summary = dspy.Example(context="This is an article.", question="This is a question?", answer="This is an answer.", rationale= "This is a rationale.").with_inputs("context", "question")

print("Example object without answer & rational keys:", article_summary.without("answer", "rationale"))
```

**Output**
```
Example object without answer & rational keys: Example({'context': 'This is an article.', 'question': 'This is a question?'}) (input_keys=None)
```

As you can see the new object returned by `without` method would not have `answer` and `rationale` keys. Access in DSPy is simple and so is the updation too!! Just how you use `.` operator to access elements you can update it's values by simply assigning it a new one!

```python
article_summary.context = "new context"
```

## Iterating over Example

A while back I said `Example` class are structurally similar dictionary what I mean is that the methods you have in dictionaries can be for in `Example` class too like `keys()`, `values()`, etc. and the way you iterate a dictionary is the same way you iterate an `Example` object. 

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