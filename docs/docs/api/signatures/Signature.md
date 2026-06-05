# dspy.Signature

A `Signature` declares the input/output contract for a language model call.
It is a Pydantic `BaseModel` whose fields are tagged as inputs or outputs
with `InputField` and `OutputField`.

There are two ways to create a signature:

**Class pattern** — recommended when you need descriptions, types, or validation:

```python
class QA(dspy.Signature):
    """Answer the question."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
```

**String pattern** — useful for rapid prototyping:

```python
QA = dspy.Signature("question -> answer")
```

All field-manipulation methods (`with_instructions`, `with_updated_fields`,
`prepend`, `append`, `insert`, `delete`) return a **new** signature class
and leave the original unchanged.

<!-- START_API_REF -->
::: dspy.Signature
    handler: python
    options:
        members:
            - append
            - delete
            - dump_state
            - equals
            - insert
            - load_state
            - prepend
            - with_instructions
            - with_updated_fields
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
<!-- END_API_REF -->
