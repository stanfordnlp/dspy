# Writing our first DSPy program

In this tutorial we’re building a haiku-generating program, extending it in each section to introduce new DSPy concepts and capabilities.

Let’s start by writing the simplest version of our program and run it, then walk through everything DSPy is doing behind the scenes:

```py
haiku_signature = "subject -> haiku"
haiku_generator = dspy.Predict(haiku_signature)
result = haiku_generator(subject="computer science")
print(result.haiku)
```

The first line specifies our `Signature`. Signature is a core concept in DSPy, and it’s how we define our task. Similar to function signatures in programming, a DSPy Signature describes the inputs a function accepts and the outputs it returns. 

The simplest way to define a DSPy Signature is with a string of the form `"inputs -> outputs"`. In our case, we want to provide a `subject` and get back a `haiku`.

Because we are programming with language models, the names of our variables matter. They both define the interface for our program and give the model a hint at our intent. If we change `haiku` to `limerick`, the model would note our cue and produce a limerick instead. Additionally, our program’s output would be accessible as `result.limerick`, rather than `result.haiku`.

To turn our Signature into a callable function, we use `dspy.Predict`. Predict is a kind of [DSPy `Module`](../diving-deeper/modules.md). If Signatures specify *what* we want, Modules define *how* we aim to achieve it. They implement a call-time strategy, manage the control flow, tools, and more. 

`Predict` is the foundational Module. Let’s look at what happens when we call `dspy.Predict(haiku_signature)`:

1. The string, `"subject -> haiku"` is parsed into a Signature class, with input and output fields defaulting to type `str`.  
2. A default instruction string is generated for the `Signature` instance, in this case: “Given the fields `subject`, produce the fields `haiku`.”  
3. A `Predict` module is instantiated with this `Signature`.

`haiku_generator` is now callable. Calling `haiku_generator(subject="computer science")` kicks off the following process:

1. The DSPy settings are checked to ensure an `LM` is configured.  
2. An [`Adapter`](../diving-deeper/adapters.md) is used to render the `Signature` and its inputs into messages the `LM` can consume. By default, this is the `ChatAdapter`, but there are JSON, XML, and other variants to format your messages suitable for a given `LM`.  
3. The `ChatAdapter` builds the prompt, which includes the `Signature` instructions, the field schema describing the inputs and outputs, formatting instructions, and the provided input (in this case, “computer science”).   
4. The messages are sent to the `LM`. Caching is enabled by default so identical calls return cached responses.  
5. A response is returned, which the `ChatAdapter` parses to extract the output fields.  
6. The `Predict` module returns a `Prediction` object with accessible output fields. Calling `result.haiku` returns the generated haiku.  
7. The call is recorded in `LM` history, which can be inspected later with `dspy.inspect_history()`.

Running `print(result.haiku)` produces:

```
Silent code unfolds
Logic threads through hidden paths
Bugs bloom, then resolve
```

In just four lines we built an AI-powered program that reads like software and acts like a function.

DSPy manages all the prompting and templating. Call `dspy.inspect_history(n=1)` to take a look at the formatted prompt our program produced and the string the `lm` returned.

First, DSPy generated the system instructions:

```
Your input fields are:
1. `subject` (str):
Your output fields are:
1. `haiku` (str):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## subject ## ]]
{subject}

[[ ## haiku ## ]]
{haiku}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
    Given the fields `subject`, produce the fields `haiku`.
```

The brackets and hashes style is how the default `ChatAdapter` structures the prompt, demarcating inputs and sections. 

Next, the adapter templated our input into a user message:

```
[[ ## subject ## ]]
computer science

Respond with the corresponding output fields, starting with the field `[[ ## haiku ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.
```

DSPy sent both to the `lm`, which produced this response:

```
[[ ## haiku ## ]]
Silent code unfolds
Logic threads through hidden paths
Bugs bloom, then resolve

[[ ## completed ## ]]
```

In DSPy, to produce a prompt you compose a signature and the messages are written for you.

---

**Next:** [Expanding signatures →](expanding-signatures.md)
