# Change inference strategies by changing the module

In our previous examples, we used the `Predict` module to execute our signature.

Other modules define different strategies for executing a task, and trying them out is very simple:

```py
reasoning_haiku_bot = dspy.ChainOfThought(HaikuBot)
result = reasoning_haiku_bot(location="Bodega Bay", mood="mysterious", season="autumn")
print(result.haiku)
```

Which yields:

```
Fog curls over waves  
Leaves whisper, gulls glide at night  
Moon pierces deep mist
```

Instead of `Predict` we used `ChainOfThought`, a module that prompts the LM to reason before delivering a final answer. Our signature, the class-based `HaikuBot`, is the same. The function call, identical.

But behind the scenes, DSPy modified our signature to prompt the model to reason before producing its final poem. The newly added signature output field, `reasoning`, is now populated by the model’s rationale.

We can view this by printing `result.reasoning`:

```
Bodega Bay in autumn evokes a cool, misty atmosphere, and the mysterious mood invites subtle, haunting imagery. I chose fog, waves, and gulls to ground the poem in the specific coastal setting, while autumnal leaves signal the season. The haiku’s 5‑7‑5 syllable structure preserves the traditional form, allowing each image to resonate briefly. The first line establishes the misty shoreline, the second line introduces quiet, nocturnal movement, and the third line offers a luminous, almost spectral image of the moon cutting through the fog—capturing mystery and the fleeting quality of autumn evenings.
```

`ChainOfThought` is simple, but nicely demonstrates the flexibility of DSPy’s module layer. [Other modules](../diving-deeper/built-in-module-variants.md) implement other strategies, including model ensembles, coding sandboxes, and *tool-calling*, which the next section explores.

---

**Next:** [Tools with ReAct →](react-and-tools.md)
