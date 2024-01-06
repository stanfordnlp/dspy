# DSPy Signatures Documentation

This documentation provides an overview of the DSPy Signatures.

## Introduction

In the DSPy framework, signatures play a crucial role in defining the behavior of language models. They are a declarative specification of the input/output behavior of a DSPy module. Instead of investing effort into _how_ to get your language model to do a sub-task, signatures enable you to inform DSPy _what_ the sub-task is. Later, the DSPy compiler will figure out how to build a complex prompt for your language model, on your data, and within your pipeline.

## Signature Class

The `Signature` class in DSPy is used to create a signature for a task. It includes the following methods:

### `__init__(self, signature: str = "", instructions: str = "")`

This method initializes a `Signature` instance. It takes a `signature` string and `instructions` string as parameters and initializes the `fields` dictionary.

### `__getattr__(self, attr)`

This method is used to get the attribute `attr` of the `Signature` instance.

### `kwargs(self)`

This property method returns a dictionary of the fields in the `Signature` instance.

### `parse_structure(self)`

This method parses the structure of the `Signature` instance, splitting the signature into input and output fields.

### `attach(self, **kwargs)`

This method attaches additional information to the fields of the `Signature` instance.

### `add_field(self, field_name: str, field_type, position="append")`

This method adds a new field to the `Signature` instance.

### `input_fields(self)`

This method returns a dictionary of the input fields in the `Signature` instance.

### `output_fields(self)`

This method returns a dictionary of the output fields in the `Signature` instance.

### `__repr__(self)`

This method returns a string representation of the `Signature` instance.

### `__eq__(self, __value: object) -> bool`

This method checks if the `Signature` instance is equal to another object.

## SignatureMeta Metaclass

The `SignatureMeta` metaclass is used to create the `Signature` class. It includes the following methods:

### `__new__(cls, name, bases, class_dict)`

This method creates a new `Signature` class.

### `kwargs(cls)`

This property method returns a dictionary of the fields in the `Signature` class.

### `__call__(cls, *args, **kwargs)`

This method calls the `Signature` class.

### `__getattr__(cls, attr)`

This method is used to get the attribute `attr` of the `Signature` class.

## Using Signatures

Here is an example of how to use signatures in DSPy:

```python
# Define a signature for a task
class MyTask(dspy.Signature):
    """This is a task."""
    input1 = dspy.InputField()
    output1 = dspy.OutputField()

# Use the signature in a module
my_module = dspy.Predict(MyTask)
```

In this example, we define a signature for a task called `MyTask`. We then use this signature in a `Predict` module.
