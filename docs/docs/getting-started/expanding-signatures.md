# Expanding your signature: more inputs and adding types

## Adding additional inputs and outputs

Adding more fields to signature strings is as easy as separating field names with commas. For example, let’s update our program to accept two inputs, a `location` and `mood`:

```py
haiku_bot = dspy.Predict("location, mood -> haiku")
result = haiku_bot(location="a quiet library", mood="mysterious")
print(result.haiku)
```

This yields:

```
Books whisper in hush  
Ink curls in soft dim corners  
Night reads in quiet
```

Defining additional outputs works identically:

```py
haiku_bot = dspy.Predict("location, mood -> haiku, haiku_title")
result = haiku_bot(location="a quiet library", mood="mysterious")
print(result.haiku_title)
print("- - -")
print(result.haiku)
```

Yielding:

```
Where the Pages Breath
- - -
Books whisper in hush  
Ink curls in soft dim corners  
Night reads in quiet
```

## Hone your signature by mindfully naming your fields

The field names we choose aren’t just for our own readability. Unlike traditional programming, where variable names are purely identifiers, the LM reads them too, and uses them to infer what each input and output means.

If we replaced `"location, mood -> haiku"` with `"a, b -> c"`, the LM would be lost. Let’s try it:

```py
haiku_bot = dspy.Predict("a, b -> c")
result = haiku_bot(a="a quiet library", b="mysterious")
print(result.c)
```

This produces:

```
In the quiet library, a mysterious presence seemed to linger between the shelves, as if it had been waiting for someone to notice.
```

Our model doesn’t know we want a haiku, so it just makes a guess.

Naming is the cheapest optimization in DSPy. A field called `research_request` will produce better completions than one called `request`, with no other changes. Signatures are easy to edit; take advantage.

## Typing your fields yields more reliable programs

We can add more specificity to our task by *typing* our fields using the format `name: type`. For example, the signature `"location, mood, contains_pun: bool -> haiku"` accepts a boolean to indicate whether we want our poem to include a pun:

```py
haiku_bot = dspy.Predict("location, mood, contains_pun: bool -> haiku")
result = haiku_bot(location="a quiet library", mood="mysterious", contains_pun=True)
print(result.haiku)
```

Which yields:

```
Dusty hush of books  
Spine turns to hush of riddles  
Novel jokes echo
```

Inline types instruct DSPy to coerce the LM’s output into the types we ask for, and surface clear warnings when they can’t. This catches a class of silent failures that prompt-only systems hide.

Types also let us communicate structural details that are easier to express in code than in natural language. [Richer types](../diving-deeper/signatures-in-depth.md) – like Pydantic models, `TypedDicts`, or `dataclasses` – can pack plenty of details that help LMs correctly complete a task.

This is especially helpful when typing output fields. For example, if we wanted to modify our program to generate several haikus we could make our output field name plural and type it as a `list[str]`:

```
haiku_bot = dspy.Predict("location, mood -> haikus:list[str]")
result = haiku_bot(location="a sunny beach", mood="relaxed")
print(f"Generated {len(result.haikus)} haikus, here's the first:")
print(result.haikus[0])
```

Which yields:

```
Generated 3 haikus, here's the first:
Sun on sandy toes,
waves breathe slow, worries drift away—
soft light, easy air.
```

Once a program accrues several fields or we want to add nuanced instructions, it’s likely time to graduate to a class-based signature.

---

**Next:** [Class-based signatures →](class-based-signatures.md)
