# Signatures in depth

## Intent

A signature is the declarative contract between your program and the language model: the input fields it accepts, the output fields it produces, and the instructions describing the task. This page covers what the class-based form gives you over strings, how typed fields move through DSPy, how to modify signatures at runtime, and the design decisions that shape it all.

Read this when you’ve outgrown the string-form `"a, b -> c"` mini-language and want to know what the class-based form adds, how typed fields are coerced, and how optimizers and modules manipulate signatures behind the scenes.

## Design decisions

### 1. Pydantic underneath

Every field on a Signature is a pydantic `FieldInfo`, and the base `Signature` class extends `pydantic.BaseModel`. DSPy-specific metadata lives in `json_schema_extra` on each field. Pydantic ships what DSPy would otherwise build by hand — type validation, JSON-schema generation, and a `TypeAdapter` that the adapter layer uses to coerce LM output into the declared type.

A useful consequence: any pydantic constraint (`gt`, `lt`, `min_length`, …) works on a field. They get stringified into `json_schema_extra["constraints"]` and the adapter mentions them in the prompt.

### 2. Dual API, one class

String form (`dspy.Signature("a, b -> c")`) and class form (`class HaikuBot(dspy.Signature)`) produce the same kind of object: a `Signature` subclass. Inline calls want a one-liner; richer specs want a docstring, typed `InputField`/`OutputField` attributes, and sometimes imported types. Collapsing both to a single class type means modules, adapters, and optimizers only ever handle one shape.

The metaclass `SignatureMeta` intercepts `dspy.Signature("...")` calls and routes them through `make_signature`. You never construct a Signature instance; the class itself is the object that gets passed around.

### 3. Docstring becomes the task instructions

The metaclass cleandocs the class docstring and stores it as `__doc__`; the `Signature.instructions` property reads it back. Absent a docstring, DSPy generates `"Given the fields X, produce the fields Y."` and uses that. The adapter renders this string into the prompt’s system message above the field schema.

Instructions belong next to the fields they describe, and optimizers need a single, well-defined string to rewrite. Treat the docstring as the place to express task intent in prose; keep field names and types short and structural, and put anything that doesn’t fit in the docstring.

### 4. Instructions live on the signature, not the module

A Signature carries its own instructions and field schema. The module (`Predict`, `ChainOfThought`, `ReAct`) supplies the call-time strategy: how many LM calls to make, what to reason about, when to stop. The same Signature can be handed to any of them, which makes it cheap to compare modules on the same task or rewrite the instructions without changing how the call is made.

If you find yourself rewriting the docstring every time you swap modules, the instructions probably contain control-flow detail that belongs in the module instead.

### 5. Immutability by deepcopy

Every mutation method (`with_instructions`, `with_updated_fields`, `prepend`, `append`, `insert`, `delete`) deep-copies the existing field dict, applies the change, and returns a new Signature class. The original stays as it was.

Optimizers run many candidate variants of the same program. If a predictor mutated its signature in place, candidates would clobber each other through shared state. Don’t try to mutate fields by reaching into `Signature.fields` directly — go through the methods. For equality, use `Signature.equals(other)`; plain `==` compares class identity, which usually isn’t what you want.

### 6. Field order is meaningful

The metaclass extracts field order from the class namespace and exposes inputs first, then outputs, via `Signature.fields`. The adapter walks that same dict when rendering the prompt, so what you see in `Signature.fields.keys()` is the order the LM sees. Reorder inputs or outputs and the prompt changes.

When inserting new fields with `prepend` / `append` / `insert`, the position argument targets the field’s own section (input or output), not the full field list.

### 7. Types are declared here but coerced elsewhere

A Signature only declares the annotation: `season: Literal["spring", ...]`, `haikus: list[str]`, a custom Pydantic model. The parsing happens in `dspy/adapters/utils.py::parse_value`, which tries `json_repair`, falls back to `ast.literal_eval`, then validates against `TypeAdapter(annotation)`. For `dspy.Type` subclasses, the adapter retries with the raw string if pydantic validation fails, letting the type’s custom parser take over.

This split exists because multiple adapters (Chat, JSON, XML, TwoStep) need to coerce the same Signature differently. JSON adapters can lean on the model’s JSON mode; XML adapters parse from tags. Keeping coercion in the adapter layer means Signatures stay lean and the parser logic isn’t duplicated. When a typed field misbehaves at parse time, look at the adapter’s output and `parse_value`, not the Signature. See [Adapters: how signatures become prompts](adapters.md).

### 8. Custom types are resolved by walking the caller’s stack frames

When the string parser sees `subject: MyType` and `MyType` isn’t a built-in or `dspy.*` class, `SignatureMeta._detect_custom_types_from_caller` walks up to 100 frames of the call stack, looking the name up in each frame’s locals and globals. This lets you write `dspy.Predict("subject: MyType -> output")` inline without importing or registering anything.

The 100-frame bound is a guardrail. In stripped or optimized Python where frames aren’t available, you’ll get a warning and need to pass `custom_types={"MyType": MyType}` to `make_signature` explicitly. The same fallback works any time you don’t want to rely on frame introspection.

### 9. Field names and descriptions are inert to optimizers

GEPA and the other instruction optimizers rewrite the Signature’s docstring. They don’t touch field names, `desc`, or `prefix`. The only way those change is `with_updated_fields`.

Field names are part of the program’s public interface: caller code reads `result.haiku`, downstream modules look up `"haiku"` by name. An optimizer that renamed them would break the program around it. So name your fields carefully — the optimizer can’t fix `result` into `haiku` later. The same goes for field descriptions when you write them.

## API walkthrough

Grouped by what you’re trying to do.

### Declaring a signature

**`dspy.Signature`**  
Subclass it once you have more than two or three fields, want a docstring, or need richer types. Calling `dspy.Signature("a, b -> c")` directly is shorthand: the metaclass intercepts the call and routes through the string parser. Both produce real Signature subclasses (not instances) — you don’t instantiate them.

**`dspy.InputField(*, desc=None, prefix=None, **pydantic_kwargs)`**  
**`dspy.OutputField(*, desc=None, prefix=None, **pydantic_kwargs)`**  
Thin factories over `pydantic.Field`. They accept every pydantic constraint (`gt`, `min_length`, …) plus the DSPy-specific `desc` and `prefix`. `prefix=` overrides the auto-inferred label; omit it and `infer_prefix` derives one from the attribute name. The deprecated `format` and `parser` args still accept values but no longer do anything — they predate the adapter layer.

**`dspy.make_signature(signature, instructions=None, signature_name="StringSignature", custom_types=None)`**  
The actual constructor. Two input shapes: a string (parsed by `_parse_signature`) or a dict of `{name: (type, FieldInfo)}` (used by the mutation methods to rebuild a class from its own pieces). `signature_name` sets the resulting class’s `__name__`, which surfaces in logs and inspection tools.

**`dspy.ensure_signature(signature, instructions=None)`**  
A no-op on already-built Signature classes; runs `make_signature` on strings. Use it when writing a module that should accept either form. It doesn’t memoize, so don’t call it in a hot loop on the same string — every call constructs a new class.

**`SignatureMeta`**  
The metaclass behind `Signature`. You won’t subclass or instantiate it. What it does: parses docstrings into instructions, validates that every field is tagged `input` or `output`, infers missing `prefix=` values, and runs the custom-type frame walk during string parsing. Most “weird signature” errors originate here.

### Introspecting a signature

**`Signature.input_fields` / `Signature.output_fields` / `Signature.fields`** → `dict[str, FieldInfo]`  
Dicts keyed by field name. The `FieldInfo`‘s `.annotation` is the declared type; `json_schema_extra` carries DSPy’s metadata, including the `__dspy_field_type` tag and the `IS_TYPE_UNDEFINED` marker (set when a string-form field has no explicit type — Predict uses it to coerce `None` to an empty string).

**`Signature.instructions`** → `str`  
The cleandoc’d docstring. The property is settable, but prefer `with_instructions(...)` so you get a fresh class.

**`Signature.signature`** → `str`  
Round-trips the class to its string form (e.g., `"location, season, mood -> haiku"`) for display. Useful in logs and inspection; not used for parsing.

**`Signature.equals(other)`** → `bool`  
Compares instructions and every field’s `json_schema_extra`. This is the equality predicate optimizers and tests use. Plain `==` compares class identity, which is usually not what you want.

### Mutating a signature

Every method here returns a new Signature class via deep-copy.

**`Signature.with_instructions(instructions: str)`**  
New class, same fields, replaced docstring.

**`Signature.with_updated_fields(name, type_=None, **json_schema_extra)`**  
Replaces the field’s metadata by merging the keyword arguments onto its existing `json_schema_extra`. Pass `desc=...` to change the description, `prefix=...` to change the label, or any custom key your adapter or optimizer reads. If you change `type_`, the new annotation replaces the old; if you also pass `IS_TYPE_UNDEFINED=False`, you clear the “treat None as empty string” behavior Predict relies on for unannotated string-form fields.

**`Signature.prepend(name, field, type_=None)`**  
**`Signature.append(name, field, type_=None)`**  
**`Signature.insert(index, name, field, type_=None)`**  
The field’s section (input or output) is determined by its `__dspy_field_type` tag — set when you built the field with `InputField` or `OutputField`. The position arg targets within that section, not across all fields. `insert` accepts negative indices; out-of-range raises `ValueError`.

**`Signature.delete(name)`**  
Silently no-ops when the field is absent. Check `name in cls.fields` first if you want loud failure.

### Persisting a signature

**`Signature.dump_state()`** → `dict`  
**`Signature.load_state(state)`** → Signature class  
Round-trip the parts that change across optimization runs: instructions, plus each field’s annotation, prefix, desc, and `__dspy_field_type` tag. The Signature class itself isn’t serialized — the assumption is that the consuming process re-instantiates it first (usually via the parent module’s saved state). `Module.save()` and `dspy.load()` use these under the hood; you rarely call them directly.

### Naming utility

**`dspy.infer_prefix(attribute_name)`** → `str`  
Splits the name on camelCase and snake_case boundaries and capitalizes the result. The metaclass calls it when a field omits `prefix=`. Rarely useful on its own.

## Cross-links

- [Adapters: how signatures become prompts](adapters.md) — what `desc`, `prefix`, and types look like on the wire, and where `parse_value` does its work.
- [Modules: composing your own](modules.md) — how `Predict`, `ChainOfThought`, and your own subclasses consume signatures.
- GEPA and the Optimizers selection guide — how the docstring gets rewritten and why field names stay fixed.
