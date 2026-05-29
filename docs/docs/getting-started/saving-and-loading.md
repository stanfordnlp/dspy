# Saving a program and reloading it

## Saving the optimized state, or the entire program

`.save(path)` has two modes. Pointing it at a `.json` file stores just the program’s learned *state*, the optimized instructions and few-shot demos, not the program’s *structure*. Pointing it at a directory with `save_program=True` writes the entire module as a pickled artifact, structure and state together.

The state-only mode is what you usually want. The JSON is small, human-readable, safe to share, and easy to diff in version control. To reload it, you instantiate your program in code as usual and call `.load()` to apply the saved state on top.

Reach for `save_program=True` when whoever is loading the program won’t have your Python class definitions handy. Shipping an optimized program to another team is one example; serving it from a different repo than the one you developed it in is another. The cost is that the saved directory contains executable Python, so only load programs from sources you trust.

```py
# state-only — small file, requires re-instantiating the program before loading
optimized_haiku_bot.save("haiku_bot.json")

# whole program — directory, rehydrates without you re-defining the class
optimized_haiku_bot.save("haiku_bot/", save_program=True)
```

## Reloading a saved program

For the directory form, `dspy.load(path)` rehydrates the full module in one call. For the state-only form, you build a fresh copy of the program in code and call `.load(path)` on it to apply the saved state.

```py
# whole program
loaded = dspy.load("haiku_bot/")

# state-only
fresh = dspy.ReAct(HaikuBot, tools=[wikipedia_search, get_wikipedia_page])
fresh.load("haiku_bot.json")
```

The save file contains the optimized instructions, demos, and signature metadata. It does not contain the LM client configuration: your API keys, your provider choice, your temperature. That separation is intentional: configure your LM as usual after loading and the same program targets whichever model you point it at today.

See [Saving and loading](../diving-deeper/saving-and-loading.md) for versioning saved programs, swapping models against the same checkpoint, and managing demo lifecycles.

---

**Next:** [Where to go next →](where-to-go-next.md)
