# Writing a class-based signature

## Adding instructional nuance with class-based signatures

A class-based signature details the same structure a string signature can, but adds a few levers for adding additional nuance. Here’s our haiku writer string signature rephrased as a class-based signature:

```py
class HaikuBot(dspy.Signature):
    """
    Write a classical haiku given the provided inputs.
    """
    location: str = dspy.InputField(desc="The setting of the poem")
    mood: str = dspy.InputField()
    haiku: str = dspy.OutputField()
```

We have the same fields (`location`, `mood`, and the `haiku` output) typed as strings, but we now have the ability to add descriptions to each. Field descriptions allow us to add nuance that might not fit within a field name.

Class-based signatures also let us write a docstring, the string at the start of the class, which DSPy uses as task instructions when preparing prompts.

We pass class-based signatures to modules just like we passed string signatures:

```py
haiku_bot = dspy.Predict(HaikuBot)
result = haiku_bot(location="a quiet library", mood="mysterious")
print(result.haiku)
```

This call renders and sends similar instructions to the LM, with two exceptions. The docstring and any field descriptions are used when building the system instructions, like so:

```
Your input fields are:
1. `location` (str): The setting of the poem
2. `mood` (str):
Your output fields are:
1. `haiku` (str):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## location ## ]]
{location}

[[ ## mood ## ]]
{mood}

[[ ## haiku ## ]]
{haiku}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
    Write a classical haiku given the provided inputs.
```

Signature docstrings and field descriptions are optional, but they are handy levers when field names don’t provide sufficient context for a task. However, resist the urge to restate what the signature already says or write prescriptive tutorials. Expansive rules, watch-outs, and guidance are what optimizers are for (more on that later). 

Though it’s worth noting: field descriptions are not touched by the optimizers, so mind your naming. A poorly chosen field name can’t be adjusted by optimizers.

## Tightening signature fields with richer types

Sometimes a plain `str` is too loose. When a value should come from a small fixed set, we’d rather pin it down so the LM (and the caller) can’t drift outside it. This is the unit-test framing from Section 3 made stricter: not just *some string*, but *one of these specific strings*.

We can use `typing`, from Python’s standard lib, to add richer types.

Typing `season` as `Literal["spring", "summer", "autumn", "winter"]` does exactly that. DSPy now accepts only those four values, both at call time and when parsing the LM’s response.

```py
from typing import Literal

Season = Literal[
    "spring", "summer", "autumn", "winter",
]

class HaikuBot(dspy.Signature):
    """
    Write a classical haiku given the provided inputs.
    """
    location: str = dspy.InputField()
    mood: str = dspy.InputField()
    season: Season = dspy.InputField()
    haiku: str = dspy.OutputField()

haiku_bot = dspy.Predict(HaikuBot)
result = haiku_bot(location="Bodega Bay", mood="mysterious", season="autumn")
print(result.haiku)
```

This yields:

```
Fog drifts over waves,  
Crimson leaves swirl by the pier—  
Night whispers secrets.
```

But if we pass `season="fall"`, we get a warning explaining the mismatch:

```
WARNING dspy.predict.predict: Type mismatch for field 'season': expected Literal['spring', 'summer', 'autumn', 'winter'] based on given Signature, but the provided value is incompatible: fall.
```

See [Signatures in depth](../diving-deeper/signatures-in-depth.md) for the rest of the surface — output validators, multi-output composition, and richer Pydantic patterns.

---

**Next:** [Changing modules →](changing-modules.md)
