---
sidebar_position: 5
---

# Data

DSPy is a machine learning framework, so working in it involves training sets, development sets, and test sets.

For each example in your data, we distinguish typically between three types of values: the inputs, the intermediate labels, and the final label. You can use DSPy effectively without any intermediate or final labels, but you will need at least a few example inputs.

## How much data do I need and how do I collect data for my task?

Concretely, you can use DSPy optimizers usefully with as few as 10 example inputs, but having 50-100 examples (or even better, 300-500 examples) goes a long way.

How can you get examples like these? If your task is extremely unusual, please invest in preparing ~10 examples by hand. Often times, depending on your metric below, you just need inputs and not labels, so it's not that hard.

However, chances are that your task is not actually that unique. You can almost always find somewhat adjacent datasets on, say, HuggingFace datasets or other forms of data that you can leverage here.

If there's data whose licenses are permissive enough, we suggest you use them. Otherwise, you can also start using/deploying/demoing your system and collect some initial data that way.

## DSPy `Example` objects

The core data type for data in DSPy is `Example`. You will use **Examples** to represent items in your training set and test set. 

DSPy **Examples** are similar to Python `dict`s but have a few useful utilities. Your DSPy modules will return values of the type `Prediction`, which is a special sub-class of `Example`.

When you use DSPy, you will do a lot of evaluation and optimization runs. Your individual datapoints will be of type `Example`:

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

Examples can have any field keys and any value types, though usually values are strings.

```text
object = Example(field1=value1, field2=value2, field3=value3, ...)
```

You can now express your training set for example as:

```python
trainset = [dspy.Example(report="LONG REPORT 1", summary="short summary 1"), ...]
```


### Specifying Input Keys

In traditional ML, there are separated "inputs" and "labels".

In DSPy, the `Example` objects have a `with_inputs()` method, which can mark specific fields as inputs. (The rest are just metadata or labels.)

```python
# Single Input.
print(qa_pair.with_inputs("question"))

# Multiple Inputs; be careful about marking your labels as inputs unless you mean it.
print(qa_pair.with_inputs("question", "answer"))
```

Values can be accessed using the `.`(dot) operator. You can access the value of key `name` in defined object `Example(name="John Doe", job="sleep")` through `object.name`. 

To access or exclude certain keys, use `inputs()` and `labels()` methods to return new Example objects containing only input or non-input keys, respectively.

```python
article_summary = dspy.Example(article= "This is an article.", summary= "This is a summary.").with_inputs("article")

input_key_only = article_summary.inputs()
non_input_key_only = article_summary.labels()

print("Example object with Input fields only:", input_key_only)
print("Example object with Non-Input fields only:", non_input_key_only)
```

**Output**
```
Example object with Input fields only: Example({'article': 'This is an article.'}) (input_keys=None)
Example object with Non-Input fields only: Example({'summary': 'This is a summary.'}) (input_keys=None)
```
