"""A signature with a custom output type, defined away from the test module that uses it.

dspy resolves a type named in a signature string by walking caller frames for the bare name. Test
modules normally define their types at module scope, which puts them on the stack and hides that
lookup's limits. Importing *this* module as a module object — `from tests.flex import
_offstack_types` — keeps `Contact` off every frame, which is the condition `bridge._resolve_signature`
exists for, and the condition any program written as `import myapp.models as models` is in.
"""

from __future__ import annotations

import pydantic

import dspy


class Contact(pydantic.BaseModel):
    name: str
    age: int


class ExtractContact(dspy.Signature):
    """Extract the contact described by the text."""

    text: str = dspy.InputField()
    contact: Contact = dspy.OutputField()
