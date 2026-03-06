"""Define DSPy signature classes and parse string-based signature specs.

This module provides the `Signature` base class, the `SignatureMeta`
metaclass, and helper functions for constructing signatures from either class
syntax or compact strings such as `"question: str, context -> answer"`.

A DSPy signature is a Pydantic model whose fields are explicitly tagged as
inputs or outputs with `InputField` and `OutputField`.  The `Signature` class
exposes convenience methods for non-mutating field and metadata manipulation
(`with_instructions`, `with_updated_fields`, `prepend`, `append`,
`insert`, `delete`) and lightweight state serialization (`dump_state`,
`load_state`).  All mutating-style methods return a **new** signature class
and leave the original unchanged.

Instruction text for each concrete signature is stored on the class's
`__doc__` attribute and exposed through the `instructions` property
defined by `SignatureMeta`.  Because of this design, the base `Signature`
class itself **cannot carry a standard class docstring** — any non-empty
docstring would be inherited by subclasses as their default instructions.
The class-level contract is therefore documented in this module docstring and
in a comment block inside the class body.

Signatures may be created in two ways:

1. **Class syntax** — define a subclass with annotated fields::

       class QA(dspy.Signature):
           '''Answer the question.'''
           question: str = dspy.InputField()
           answer: str = dspy.OutputField()

2. **String syntax** — call the base class to get a generated subclass::

       QA = dspy.Signature("question -> answer")

In the class syntax the subclass's docstring becomes the instruction text.
In the string syntax, instructions can be supplied as the second positional
argument or via the `instructions` keyword argument.

In addition to class-based definitions, this module supports:

- constructing signatures from strings or field mappings via `make_signature`
- normalizing string-or-signature inputs via `ensure_signature`
- resolving user-defined annotation names that appear in string signatures
- creating modified copies of existing signatures without mutating the
  original class
"""

import ast
import importlib
import inspect
import re
import sys
import types
import typing
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

from dspy.signatures.field import InputField, OutputField


def _default_instructions(cls) -> str:
    inputs_ = ", ".join([f"`{field}`" for field in cls.input_fields])
    outputs_ = ", ".join([f"`{field}`" for field in cls.output_fields])
    return f"Given the fields {inputs_}, produce the fields {outputs_}."


class SignatureMeta(type(BaseModel)):
    """Control how DSPy signature classes are created and called.

    This metaclass is responsible for two distinct behaviors:

    1. Calling the base `Signature` class, as in `Signature("x -> y")`, does
       not create an instance. Instead, it constructs and returns a new
       `Signature` subclass.
    2. Creating any concrete `Signature` subclass validates that all model
       fields are tagged as DSPy input or output fields, preserves field order,
       fills in default annotations where needed, and ensures required DSPy
       metadata such as `prefix` and `desc` is present.

    The result is a Pydantic-backed class that also serves as DSPy's signature
    schema object.
    """

    def __call__(cls, *args, **kwargs):
        """Create either a new signature class or a signature instance.

        Calling the base `Signature` class is special: `Signature("input -> output")`
        returns a newly generated `Signature` subclass instead of an instance.
        Calling a concrete subclass such as `MySignature(...)` uses normal
        Pydantic model construction and returns an instance of that subclass.

        If the caller passes a string signature and does not explicitly provide
        `custom_types`, this method tries to resolve user-defined annotation
        names from the caller's stack frame before delegating to
        `make_signature`.  Only top-level annotation names (the name
        immediately after `:` in each field spec) are auto-detected; names
        nested inside generic brackets, such as `Passage` in
        `list[Passage]`, require explicit `custom_types`.

        Args:
            *args: Positional arguments forwarded to `make_signature` when
                calling `Signature`, or to normal model construction when
                calling a concrete signature subclass.
            **kwargs: Keyword arguments forwarded to `make_signature` when
                calling `Signature`, or to normal model construction when
                calling a concrete signature subclass.

        Returns:
            type[Signature] | Signature: A newly generated signature class when
            called on `Signature`, or an instance when called on a concrete
            signature subclass.

        Examples:
            `Signature("question -> answer")` returns a new signature class.
            `MySignature(question="...")` returns a `MySignature` instance.
        """
        if cls is Signature:
            # We don't create an actual Signature instance, instead, we create a new Signature class.
            custom_types = kwargs.pop("custom_types", None)

            if custom_types is None and args and isinstance(args[0], str):
                custom_types = cls._detect_custom_types_from_caller(args[0])

            return make_signature(*args, custom_types=custom_types, **kwargs)
        return super().__call__(*args, **kwargs)

    @staticmethod
    def _detect_custom_types_from_caller(signature_str):
        """Infer annotation bindings for a string signature from caller frames.

        String-based signatures can resolve built-in types and names from
        `typing` directly, but user-defined annotations such as `Passage`
        in `"context: Passage -> answer"` need a runtime object to bind to
        that name.  This helper extracts candidate type names from
        `signature_str` and walks up the call stack looking for matching
        names in local or global scope.

        The extraction uses a regex that captures the first name token after
        each `:` in the signature string, including dotted names like
        `Module.Type`.  This means only **top-level annotation names** are
        detected.  Names nested inside generic brackets — for example,
        `Passage` in `list[Passage]` — are **not** extracted and therefore
        cannot be auto-resolved by this method.

        The returned mapping is suitable for the `custom_types` argument of
        `make_signature`.  It is used only while parsing the string signature
        and is not stored on the resulting signature class.

        This lookup is a convenience, not a guarantee.  It may fail when frame
        introspection is unavailable, when the relevant type is not present in
        the caller's namespace, or when the desired type appears deeper in the
        stack than the search limit.

        Args:
            signature_str: String-form signature that may contain explicit type
                annotations, for example `"question: Passage -> answer"` or
                `"query: Module.CustomType -> answer"`.

        Returns:
            dict[str, Any] | None: Mapping from annotation names to runtime
            Python objects, or `None` if no user-defined annotation names
            were found.

        Note:
            For predictable behavior, callers that depend on user-defined
            annotation types should pass `custom_types` explicitly instead
            of relying on stack introspection.  This is especially important
            for types nested inside generics, for example::

                Signature(
                    "context: list[Passage] -> answer",
                    custom_types={"Passage": Passage},
                )
        """

        # Extract potential type names from the signature string, including dotted names
        # Match both simple types like 'MyType' and dotted names like 'Module.Type'
        type_pattern = r":\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)"
        type_names = re.findall(type_pattern, signature_str)
        if not type_names:
            return None

        # Get type references from caller frames by walking the stack
        found_types = {}

        needed_types = set()
        dotted_types = {}

        for type_name in type_names:
            parts = type_name.split(".")
            base_name = parts[0]

            if base_name not in typing.__dict__ and base_name not in __builtins__:
                if len(parts) > 1:
                    dotted_types[type_name] = base_name
                    needed_types.add(base_name)
                else:
                    needed_types.add(type_name)

        if not needed_types:
            return None

        frame = None
        try:
            frame = sys._getframe(1)  # Start one level up (skip this function)

            max_frames = 100
            frame_count = 0

            while frame and needed_types and frame_count < max_frames:
                frame_count += 1

                for type_name in list(needed_types):
                    if type_name in frame.f_locals:
                        found_types[type_name] = frame.f_locals[type_name]
                        needed_types.remove(type_name)
                    elif frame.f_globals and type_name in frame.f_globals:
                        found_types[type_name] = frame.f_globals[type_name]
                        needed_types.remove(type_name)

                # If we found all needed types, stop looking
                if not needed_types:
                    break

                frame = frame.f_back

            if needed_types and frame_count >= max_frames:
                import logging

                logging.getLogger("dspy").warning(
                    f"Reached maximum frame search depth ({max_frames}) while looking for types: {needed_types}. "
                    "Consider providing custom_types explicitly to Signature."
                )
        except (AttributeError, ValueError):
            # Handle environments where frame introspection is not available
            import logging

            logging.getLogger("dspy").debug(
                "Frame introspection failed while trying to resolve custom types. "
                "Consider providing custom_types explicitly to Signature."
            )
        finally:
            if frame:
                del frame

        return found_types or None

    def __new__(mcs, signature_name, bases, namespace, **kwargs):
        """Create a new signature class and normalize its field metadata.

        During class creation this method:

        1. Preserves declaration order for DSPy fields.
        2. Assigns `str` as the default annotation for any field declared with
           `InputField` or `OutputField` but without an explicit type.
        3. Lets Pydantic build the actual model class.
        4. Determines instruction text by inheriting a base signature's
           instructions when appropriate or synthesizing default instructions
           from the field names.
        5. Verifies that every field is explicitly marked as an input or output
           field.
        6. Ensures every field has default DSPy metadata for `prefix` and
           `desc`.

        Args:
            mcs: Metaclass object.
            signature_name: Name of the class being created.
            bases: Base classes for the new class.
            namespace: Class body namespace.
            **kwargs: Additional keyword arguments forwarded to Pydantic's class
                construction machinery.

        Returns:
            type[Signature]: Newly created signature subclass.

        Raises:
            TypeError: If any declared model field is not tagged with
                `InputField` or `OutputField` metadata.
        """
        # At this point, the orders have been swapped already.
        field_order = [name for name, value in namespace.items() if isinstance(value, FieldInfo)]
        # Set `str` as the default type for all fields
        if sys.version_info >= (3, 14):
            try:
                import annotationlib
                # Try to get from explicit __annotations__ first (e.g., from __future__ import annotations)
                raw_annotations = namespace.get("__annotations__")

                if raw_annotations is None:
                    # In 3.14 with PEP 649, get the annotate function and call it
                    annotate_func = annotationlib.get_annotate_from_class_namespace(namespace)
                    if annotate_func:
                        raw_annotations = annotationlib.call_annotate_function(
                            annotate_func,
                            format=annotationlib.Format.FORWARDREF
                        )
                    else:
                        raw_annotations = {}
            except ImportError:
                raw_annotations = namespace.get("__annotations__", {})
        else:
            # Python 3.13 and earlier
            # Set `str` as the default type for all fields
            raw_annotations = namespace.get("__annotations__", {})
        for name, field in namespace.items():
            if not isinstance(field, FieldInfo):
                continue  # Don't add types to non-field attributes
            if not name.startswith("__") and name not in raw_annotations:
                raw_annotations[name] = str
        # Create ordered annotations dictionary that preserves field order
        ordered_annotations = {name: raw_annotations[name] for name in field_order if name in raw_annotations}
        # Add any remaining annotations that weren't in field_order
        ordered_annotations.update({k: v for k, v in raw_annotations.items() if k not in ordered_annotations})
        namespace["__annotations__"] = ordered_annotations

        # Let Pydantic do its thing
        cls = super().__new__(mcs, signature_name, bases, namespace, **kwargs)

        # If we don't have instructions, it might be because we are a derived generic type.
        # In that case, we should inherit the instructions from the base class.
        if cls.__doc__ is None:
            for base in bases:
                if isinstance(base, SignatureMeta):
                    doc = getattr(base, "__doc__", "")
                    if doc != "":
                        cls.__doc__ = doc

        # The more likely case is that the user has just not given us a type.
        # In that case, we should default to the input/output format.
        if cls.__doc__ is None:
            cls.__doc__ = _default_instructions(cls)

        # Ensure all fields are declared with InputField or OutputField
        cls._validate_fields()

        # Ensure all fields have a prefix
        for name, field in cls.model_fields.items():
            if "prefix" not in field.json_schema_extra:
                field.json_schema_extra["prefix"] = infer_prefix(name) + ":"
            if "desc" not in field.json_schema_extra:
                field.json_schema_extra["desc"] = f"${{{name}}}"

        return cls

    def _validate_fields(cls):
        """Ensure every model field is tagged as a DSPy input or output field.

        DSPy signatures rely on field metadata stored under
        `field.json_schema_extra["__dspy_field_type"]` to distinguish inputs
        from outputs. This validator enforces that invariant at class creation
        time so later helpers can safely partition fields without additional
        checks.

        Raises:
            TypeError: If any field is missing the DSPy field-type marker or if
                the marker is not `"input"` or `"output"`.
        """
        for name, field in cls.model_fields.items():
            extra = field.json_schema_extra or {}
            field_type = extra.get("__dspy_field_type")
            if field_type not in ["input", "output"]:
                raise TypeError(
                    f"Field `{name}` in `{cls.__name__}` must be declared with InputField or OutputField, but "
                    f"field `{name}` has `field.json_schema_extra={field.json_schema_extra}`",
                )

    @property
    def instructions(cls) -> str:
        """Return the signature's instruction text with normalized indentation.

        DSPy stores a signature's natural-language instructions in the class
        docstring. This property exposes that text in a clean form by applying
        `inspect.cleandoc`.

        Returns:
            str: Instruction text associated with the signature class.
        """
        return inspect.cleandoc(getattr(cls, "__doc__", ""))

    @instructions.setter
    def instructions(cls, instructions: str) -> None:
        """Replace the instruction text stored on the signature class.

        Args:
            instructions: New instruction text to assign to `cls.__doc__`.
        """
        cls.__doc__ = instructions

    @property
    def input_fields(cls) -> dict[str, FieldInfo]:
        """Return the signature's input fields in declaration order.

        Returns:
            dict[str, FieldInfo]: Mapping from field name to field definition for
            all fields tagged as DSPy input fields.
        """
        return cls._get_fields_with_type("input")

    @property
    def output_fields(cls) -> dict[str, FieldInfo]:
        """Return the signature's output fields in declaration order.

        Returns:
            dict[str, FieldInfo]: Mapping from field name to field definition for
            all fields tagged as DSPy output fields.
        """
        return cls._get_fields_with_type("output")

    @property
    def fields(cls) -> dict[str, FieldInfo]:
        """Return all signature fields with inputs ordered before outputs.

        The returned mapping preserves declaration order within the input and
        output sections, then concatenates those sections as
        `{**cls.input_fields, **cls.output_fields}`.

        Returns:
            dict[str, FieldInfo]: Combined mapping of all fields.
        """
        # Make sure to give input fields before output fields
        return {**cls.input_fields, **cls.output_fields}

    @property
    def signature(cls) -> str:
        """Return the compact string form of the signature.

        The string contains only field names, not type annotations or
        descriptions, and is formatted as `"input1, input2 -> output1, output2"`.

        Returns:
            str: Canonical name-only string representation of the signature.
        """
        input_fields = ", ".join(cls.input_fields.keys())
        output_fields = ", ".join(cls.output_fields.keys())
        return f"{input_fields} -> {output_fields}"

    def _get_fields_with_type(cls, field_type) -> dict[str, FieldInfo]:
        """Return the subset of model fields tagged with a given DSPy field type.

        Args:
            field_type: DSPy field-type tag to filter by, typically `"input"`
                or `"output"`.

        Returns:
            dict[str, FieldInfo]: Filtered mapping of fields whose
            `json_schema_extra["__dspy_field_type"]` equals `field_type`.
        """
        return {k: v for k, v in cls.model_fields.items() if v.json_schema_extra["__dspy_field_type"] == field_type}

    def __repr__(cls):
        """Return a debug-friendly representation of the signature class.

        The representation includes the class name, compact signature string,
        normalized instructions, and one `Field(...)` entry per declared DSPy
        field.

        Returns:
            str: Multiline representation intended for debugging and interactive
            inspection.
        """
        field_reprs = []
        for name, field in cls.fields.items():
            field_reprs.append(f"{name} = Field({field})")
        field_repr = "\n    ".join(field_reprs)
        return f"{cls.__name__}({cls.signature}\n    instructions={cls.instructions!r}\n    {field_repr}\n)"


class Signature(BaseModel, metaclass=SignatureMeta):
    ""
    # Design constraint: Signature cannot carry a standard class docstring.
    #
    # DSPy stores each concrete signature's instruction text in `cls.__doc__`.
    # A non-empty docstring here would be inherited by every subclass that does
    # not define its own, incorrectly overriding the default instructions that
    # `SignatureMeta.__new__` synthesizes from field names.  See the module
    # docstring above for the full class-level contract.
    #
    # In brief, a `Signature` subclass is a Pydantic `BaseModel` whose
    # fields are tagged with `InputField` or `OutputField`.  Every signature
    # class also carries instruction text (the `instructions` property on
    # `SignatureMeta`) and convenience methods for non-mutating manipulation
    # of fields and metadata.
    #
    # Signatures may be created via class syntax::
    #
    #     class QA(dspy.Signature):
    #         '''Answer the question.'''
    #         question: str = dspy.InputField()
    #         answer: str = dspy.OutputField()
    #
    # or via string syntax::
    #
    #     QA = dspy.Signature("question -> answer")
    #
    # All field-manipulation methods (`with_instructions`,
    # `with_updated_fields`, `prepend`, `append`, `insert`, `delete`)
    # return a **new** signature class and leave the original unchanged.
    #
    # `dump_state` / `load_state` serialize and restore only the mutable
    # prompt-label metadata (instructions, per-field `prefix` and `desc`).
    # Field names, types, validators, and other Pydantic schema details are not
    # included in the serialized state.

    @classmethod
    def with_instructions(cls, instructions: str) -> type["Signature"]:
        """Return a new signature class with identical fields and new instructions.

        This helper is a non-mutating way to reword a signature's
        natural-language contract. Field order, types, and DSPy field metadata are
        copied from `cls` unchanged.

        Args:
            instructions: Instruction text to attach to the new signature class.

        Returns:
            type[Signature]: Fresh signature subclass whose fields match
            `cls.fields` and whose `instructions` equal `instructions`.

        Examples:
            >>> import dspy
            >>> class MySig(dspy.Signature):
            ...     input_text: str = dspy.InputField(desc="Input text")
            ...     output_text: str = dspy.OutputField(desc="Output text")
            >>> NewSig = MySig.with_instructions("Translate to French.")
            >>> NewSig is MySig
            False
            >>> NewSig.instructions
            'Translate to French.'
        """
        return Signature(cls.fields, instructions)

    @classmethod
    def with_updated_fields(cls, name: str, type_: type | None = None, **kwargs: dict[str, Any]) -> type["Signature"]:
        """Create a copy of the signature with one field's metadata updated.

        This method deep-copies the current field mapping, merges `kwargs` into
        the target field's `json_schema_extra`, optionally replaces the field
        annotation with `type_`, and returns a fresh signature class. The
        original signature class is not mutated.

        Args:
            name: Name of the field to update.
            type_: Optional replacement type annotation for the field. If
                omitted, the existing annotation is preserved.
            **kwargs: Metadata entries to merge into the field's
                `json_schema_extra`. On key conflicts, values from `kwargs`
                override the existing metadata.

        Returns:
            type[Signature]: New signature class with the updated field
            metadata.

        Raises:
            KeyError: If `name` is not a field on the current signature.
        """
        fields_copy = deepcopy(cls.fields)
        # Update `fields_copy[name].json_schema_extra` with the new kwargs, on conflicts
        # we use the new value in kwargs.
        fields_copy[name].json_schema_extra = {
            **fields_copy[name].json_schema_extra,
            **kwargs,
        }
        if type_ is not None:
            fields_copy[name].annotation = type_
        return Signature(fields_copy, cls.instructions)

    @classmethod
    def prepend(cls, name, field, type_=None) -> type["Signature"]:
        """Insert a field at the start of its input or output section.

        The target section is determined by `field.json_schema_extra`.
        Prepending an input field places it before all existing inputs.
        Prepending an output field places it before all existing outputs. The
        original signature class is left unchanged.

        Args:
            name: Field name to add.
            field: `InputField` or `OutputField` instance to insert.
            type_: Optional explicit type annotation. If omitted, `insert`
                first uses `field.annotation` and then falls back to `str`.

        Returns:
            type[Signature]: New signature class with the field inserted first
            within its section.

        Examples:
            >>> import dspy
            >>> class MySig(dspy.Signature):
            ...     input_text: str = dspy.InputField(desc="Input sentence")
            ...     output_text: str = dspy.OutputField(desc="Translated sentence")
            >>> NewSig = MySig.prepend("context", dspy.InputField(desc="Context for translation"))
            >>> list(NewSig.fields.keys())
            ['context', 'input_text', 'output_text']
        """
        return cls.insert(0, name, field, type_)

    @classmethod
    def append(cls, name, field, type_=None) -> type["Signature"]:
        """Insert a field at the end of its input or output section.

        The target section is determined by `field.json_schema_extra`.
        Appending an input field places it after the existing inputs but before
        any outputs. Appending an output field places it after the existing
        outputs. The original signature class is left unchanged.

        Args:
            name: Field name to add.
            field: `InputField` or `OutputField` instance to insert.
            type_: Optional explicit type annotation. If omitted, `insert`
                first uses `field.annotation` and then falls back to `str`.

        Returns:
            type[Signature]: New signature class with the field appended within
            its section.

        Examples:
            >>> import dspy
            >>> class MySig(dspy.Signature):
            ...     input_text: str = dspy.InputField(desc="Input sentence")
            ...     output_text: str = dspy.OutputField(desc="Translated sentence")
            >>> NewSig = MySig.append("confidence", dspy.OutputField(desc="Translation confidence"))
            >>> list(NewSig.fields.keys())
            ['input_text', 'output_text', 'confidence']
        """
        return cls.insert(-1, name, field, type_)

    @classmethod
    def delete(cls, name) -> type["Signature"]:
        """Return a new signature class without one field.

        If `name` is not present, this method returns a new signature whose
        fields are unchanged. No error is raised for missing field names.

        Args:
            name: Field name to remove.

        Returns:
            type[Signature]: New signature class with the field removed, or an
            equivalent copy if the field was absent.

        Examples:
            >>> import dspy
            >>> class MySig(dspy.Signature):
            ...     input_text: str = dspy.InputField(desc="Input sentence")
            ...     temp_field: str = dspy.InputField(desc="Temporary debug field")
            ...     output_text: str = dspy.OutputField(desc="Translated sentence")
            >>> NewSig = MySig.delete("temp_field")
            >>> list(NewSig.fields.keys())
            ['input_text', 'output_text']
            >>> list(NewSig.delete("nonexistent").fields.keys())
            ['input_text', 'output_text']
        """
        fields = dict(cls.fields)

        fields.pop(name, None)

        return Signature(fields, cls.instructions)

    @classmethod
    def insert(cls, index: int, name: str, field, type_: type | None = None) -> type["Signature"]:
        """Insert a field at a specific position within its input or output section.

        The section is determined by `field.json_schema_extra["__dspy_field_type"]`.
        Input fields are inserted among inputs and output fields are inserted
        among outputs. The resulting signature still stores all inputs before
        all outputs.

        Negative indices are supported.  An index of `-1` appends within the
        chosen section.  If `type_` is omitted, this method first uses
        `field.annotation` and then falls back to `str`.

        Args:
            index: Insertion position within the chosen section.  Negative
                values are interpreted relative to the section length plus one.
            name: Field name to add.
            field: `InputField` or `OutputField` instance to insert.  Must
                carry the DSPy field-type marker in
                `field.json_schema_extra["__dspy_field_type"]`.
            type_: Optional explicit type annotation for the inserted field.

        Returns:
            type[Signature]: New signature class with the field inserted.

        Raises:
            ValueError: If `index` falls outside the valid insertion range
                for the chosen section.
            KeyError: If `field.json_schema_extra` does not contain the
                `"__dspy_field_type"` key.
            TypeError: If `field.json_schema_extra` is `None` (i.e. the
                field was not created via `InputField` or `OutputField`).

        Examples:
            >>> import dspy
            >>> class MySig(dspy.Signature):
            ...     input_text: str = dspy.InputField(desc="Input sentence")
            ...     output_text: str = dspy.OutputField(desc="Translated sentence")
            >>> NewSig = MySig.insert(0, "context", dspy.InputField(desc="Context for translation"))
            >>> list(NewSig.fields.keys())
            ['context', 'input_text', 'output_text']
            >>> NewSig2 = NewSig.insert(-1, "confidence", dspy.OutputField(desc="Translation confidence"))
            >>> list(NewSig2.fields.keys())
            ['context', 'input_text', 'output_text', 'confidence']
        """
        # It's possible to set the type as annotation=type in pydantic.Field(...)
        # But this may be annoying for users, so we allow them to pass the type
        if type_ is None:
            type_ = field.annotation
        if type_ is None:
            type_ = str

        input_fields = list(cls.input_fields.items())
        output_fields = list(cls.output_fields.items())

        # Choose the list to insert into based on the field type
        lst = input_fields if field.json_schema_extra["__dspy_field_type"] == "input" else output_fields
        # We support negative insert indices
        if index < 0:
            index += len(lst) + 1
        if index < 0 or index > len(lst):
            raise ValueError(
                f"Invalid index to insert: {index}, index must be in the range of [{len(lst) - 1}, {len(lst)}] for "
                f"{field.json_schema_extra['__dspy_field_type']} fields, but received: {index}.",
            )
        lst.insert(index, (name, (type_, field)))

        new_fields = dict(input_fields + output_fields)
        return Signature(new_fields, cls.instructions)

    @classmethod
    def equals(cls, other) -> bool:
        """Compare this signature's instructions and field metadata with another.

        This comparison is intentionally narrower than full Pydantic model
        equality. It requires that `other` be a `BaseModel` subclass, that both
        signatures have identical `instructions`, and that every field name be
        present on both sides with identical `json_schema_extra` metadata.

        The method does not compare field type annotations, defaults,
        validators, or other schema details beyond the DSPy metadata stored in
        `json_schema_extra`.

        Args:
            other: Candidate signature class to compare against.  Should be a
                `Signature` subclass (or at least use `SignatureMeta`);
                non-`Signature` `BaseModel` subclasses pass the initial
                type check but will raise `AttributeError` when
                `instructions` or `fields` is accessed.

        Returns:
            bool: `True` if the two signatures match under this
            metadata-based comparison, otherwise `False`.
        """
        if not isinstance(other, type) or not issubclass(other, BaseModel):
            return False
        if cls.instructions != other.instructions:
            return False
        for name in cls.fields.keys() | other.fields.keys():
            if name not in other.fields or name not in cls.fields:
                return False
            if cls.fields[name].json_schema_extra != other.fields[name].json_schema_extra:
                return False
        return True

    @classmethod
    def dump_state(cls):
        """Serialize the mutable instruction and prompt-label state of a signature.

        The dumped state is intentionally lightweight.  It captures only the
        signature's instruction text and, for each field in current field
        order, the DSPy `prefix` and `desc` metadata.  Field types, field
        names, validators, and other Pydantic schema details are not included.

        The serialized key for the description is `"description"` (not
        `"desc"`), matching the convention used by `load_state`.

        Returns:
            dict: JSON-serializable state with the structure::

                {
                    "instructions": str,
                    "fields": [
                        {"prefix": str, "description": str},
                        ...
                    ],
                }
        """
        state = {"instructions": cls.instructions, "fields": []}
        for field in cls.fields:
            state["fields"].append(
                {
                    "prefix": cls.fields[field].json_schema_extra["prefix"],
                    "description": cls.fields[field].json_schema_extra["desc"],
                }
            )

        return state

    @classmethod
    def load_state(cls, state):
        """Apply serialized instruction and field-label state to a copy of the signature.

        This method deep-copies the current signature's fields, builds a new
        signature class from them, and then overwrites the clone's instructions
        and per-field `prefix` and `desc` metadata using `state`.  Field
        updates are applied by zipping the saved field entries against the
        current field order.  The original signature class is **not** mutated.

        Args:
            state: State dict previously produced by `dump_state`.  Must
                contain an `"instructions"` string and a `"fields"` list
                where each entry is a dict with `"prefix"` and
                `"description"` keys.

        Returns:
            type[Signature]: New signature class with updated instructions and
            field labels.

        Raises:
            KeyError: If `state` is missing expected top-level keys
                (`"instructions"`, `"fields"`) or if any field entry is
                missing `"prefix"` or `"description"`.

        Note:
            `state["fields"]` is matched **positionally** against the
            current field order, not by field name.  If the saved state has
            fewer field entries than the signature, remaining fields keep
            their existing metadata.  Extra saved entries are silently
            ignored.
        """
        signature_copy = Signature(deepcopy(cls.fields), cls.instructions)

        signature_copy.instructions = state["instructions"]
        for field, saved_field in zip(signature_copy.fields.values(), state["fields"], strict=False):
            field.json_schema_extra["prefix"] = saved_field["prefix"]
            field.json_schema_extra["desc"] = saved_field["description"]

        return signature_copy


def ensure_signature(signature: str | type[Signature], instructions=None) -> type[Signature]:
    """Normalize a string or signature-like value into a signature class.

    This helper is useful at API boundaries where callers may supply a string
    signature such as `"question -> answer"`, an existing signature class,
    or `None`.

    Strings are parsed by calling `Signature(signature, instructions)`.
    `None` is returned unchanged.  Any other value is returned unchanged as
    long as `instructions` is not also provided.

    Args:
        signature: `None`, a string signature, or an existing
            signature-like value.
        instructions: Optional instructions used only when `signature` is a
            string.

    Returns:
        type[Signature] | None: Parsed signature class, the original
        non-string value unchanged, or `None`.

    Raises:
        ValueError: If `instructions` is provided together with a
            non-string `signature` value.

    Note:
        This helper does not validate that non-string inputs are actually
        `Signature` subclasses.  Callers are responsible for ensuring type
        correctness when passing non-string values.
    """
    if signature is None:
        return None
    if isinstance(signature, str):
        return Signature(signature, instructions)
    if instructions is not None:
        raise ValueError("Don't specify instructions when initializing with a Signature")
    return signature


def make_signature(
    signature: str | dict[str, tuple[type, FieldInfo]],
    instructions: str | None = None,
    signature_name: str = "StringSignature",
    custom_types: dict[str, type] | None = None,
) -> type[Signature]:
    """Create a `Signature` subclass from a string spec or field mapping.

    This function is the constructor behind `Signature("...")`. It validates
    the supplied fields, synthesizes default instructions when needed, and
    returns a new Pydantic model class derived from `Signature`.

    When `signature` is a string, explicit annotations inside that string are
    parsed with the helpers in this module. `custom_types` can be supplied as
    an annotation-resolution map from type names in the string to runtime
    Python objects.

    `custom_types` is used only while parsing a string signature. It is not the
    same concept as adapter-level DSPy custom types such as subclasses of
    `dspy.adapters.types.Type`, although those classes may also appear in the
    mapping when they are referenced by name in the string.

    Args:
        signature: Either a string in the form
            `"input1, input2 -> output1, output2"` or a mapping from field
            names to `(type, FieldInfo)` tuples. Each mapping value may also be
            a bare `FieldInfo` whose `annotation` is already set.
        instructions: Instruction text to attach to the generated signature. If
            omitted, default instructions are synthesized from the input and
            output field names.
        signature_name: Class name to use for the generated signature subclass.
        custom_types: Optional mapping from type names used in a string
            signature to runtime Python objects used to resolve those
            annotations. For example, pass `{"Passage": Passage}` to parse
            `list[Passage]`. This mapping is ignored when `signature` is already
            a field mapping; downstream DSPy code uses the resolved field
            annotations, not this dictionary itself.

    Returns:
        type[Signature]: Newly created signature subclass.

    Raises:
        ValueError: If the signature string is malformed, a field name is not a
            string, a field specification is malformed, a field type is invalid,
            or a field value is not a `FieldInfo` instance.
        SyntaxError: If a string signature contains syntax that cannot be
            parsed as a Python parameter list (raised by the underlying
            `ast.parse` call in `_parse_field_string`).
        TypeError: If any resulting field is not tagged with `InputField` or
            `OutputField` metadata (raised during class creation by
            `SignatureMeta._validate_fields`).

    Examples:
        Create a signature from a simple string:

        >>> sig1 = make_signature("question, context -> answer")
        >>> sig1.signature
        'question, context -> answer'

        Create a signature from an explicit field mapping:

        >>> sig2 = make_signature({
        ...     "question": (str, InputField(desc="Question to answer")),
        ...     "answer": (str, OutputField(desc="Final answer")),
        ... })
        >>> list(sig2.fields.keys())
        ['question', 'answer']

        Use all arguments, including explicit instructions, a custom generated
        class name, and an annotation-resolution mapping for a user-defined
        type referenced in the string signature:

        >>> class Passage:
        ...     pass
        >>> sig3 = make_signature(
        ...     signature="question: str, context: list[Passage] -> answer: str",
        ...     instructions="Answer the question using the provided context.",
        ...     signature_name="QaWithPassages",
        ...     custom_types={"Passage": Passage},
        ... )
        >>> sig3.__name__
        'QaWithPassages'
        >>> sig3.instructions
        'Answer the question using the provided context.'
        >>> sig3.signature
        'question, context -> answer'
    """
    # Prepare the names dictionary for type resolution
    names = None
    if custom_types:
        names = dict(typing.__dict__)
        names.update(custom_types)

    fields = _parse_signature(signature, names) if isinstance(signature, str) else signature

    # Validate the fields, this is important because we sometimes forget the
    # slightly unintuitive syntax with tuples of (type, Field)
    fixed_fields = {}
    for name, type_field in fields.items():
        if not isinstance(name, str):
            raise ValueError(f"Field names must be strings, but received: {name}.")
        if isinstance(type_field, FieldInfo):
            type_ = type_field.annotation
            field = type_field
        else:
            if not isinstance(type_field, tuple):
                raise ValueError(f"Field values must be tuples, but received: {type_field}.")
            type_, field = type_field
        # It might be better to be explicit about the type, but it currently would break
        # program of thought and teleprompters, so we just silently default to string.
        if type_ is None:
            type_ = str
        if not isinstance(
            type_, (type, typing._GenericAlias, types.GenericAlias, typing._SpecialForm, types.UnionType)
        ):
            raise ValueError(f"Field types must be types, but received: {type_} of type {type(type_)}.")
        if not isinstance(field, FieldInfo):
            raise ValueError(f"Field values must be Field instances, but received: {field}.")
        fixed_fields[name] = (type_, field)

    # Default prompt when no instructions are provided
    if instructions is None:
        sig = Signature(signature, "")  # Simple way to parse input/output fields
        instructions = _default_instructions(sig)

    return create_model(
        signature_name,
        __base__=Signature,
        __doc__=instructions,
        **fixed_fields,
    )


def _parse_signature(signature: str, names=None) -> dict[str, tuple[type, Field]]:
    """Parse a string signature into DSPy field definitions.

    The input string must contain exactly one `->` separator.  Fields before
    the separator become `InputField` definitions, and fields after the
    separator become `OutputField` definitions.  Each side is parsed using
    Python's parameter-list syntax, so annotations such as `x: int` or
    `items: list[str]` are supported.

    If a field name appears on both sides of `->`, the output-side
    definition silently overwrites the input-side definition.  Duplicate
    names within the same side are also resolved by last-write-wins.

    Args:
        signature: String-form signature such as
            `"question: str, context: list[str] -> answer: str"`.
        names: Optional mapping of annotation names to runtime objects used
            while resolving explicit type annotations.

    Returns:
        dict[str, tuple[type, FieldInfo]]: Mapping from field name to
        `(annotation, FieldInfo)` pairs suitable for `make_signature`.

    Raises:
        ValueError: If `signature` does not contain exactly one `->`,
            or if an annotation name cannot be resolved by
            `_parse_type_node`.
        SyntaxError: If either side of the signature is not valid Python
            parameter-list syntax (raised by `ast.parse` inside
            `_parse_field_string`).
    """
    if signature.count("->") != 1:
        raise ValueError(f"Invalid signature format: '{signature}', must contain exactly one '->'.")

    inputs_str, outputs_str = signature.split("->")

    fields = {}
    for field_name, field_type in _parse_field_string(inputs_str, names):
        fields[field_name] = (field_type, InputField())
    for field_name, field_type in _parse_field_string(outputs_str, names):
        fields[field_name] = (field_type, OutputField())

    return fields


def _parse_field_string(field_string: str, names=None) -> dict[str, str]:
    """Parse one side of a string signature into ordered field/type pairs.

    This helper reuses Python's own parser by embedding `field_string` inside
    a temporary function signature.  For example, `"x: int, y: str"` is
    parsed as though it were the parameter list of `def f(...): pass`.

    Args:
        field_string: Comma-separated field fragment from the left or right
            side of a DSPy string signature.
        names: Optional mapping of annotation names to runtime objects used
            while resolving explicit type annotations via
            `_parse_type_node`.

    Returns:
        Iterable[tuple[str, Any]]: Ordered `(field_name, field_type)` pairs.
        Fields without explicit annotations default to `str`.

    Raises:
        SyntaxError: If `field_string` is not valid Python parameter-list
            syntax (raised by `ast.parse`).
        ValueError: If any annotation cannot be resolved by
            `_parse_type_node`.
    """

    args = ast.parse(f"def f({field_string}): pass").body[0].args.args
    field_names = [arg.arg for arg in args]
    types = [str if arg.annotation is None else _parse_type_node(arg.annotation, names) for arg in args]
    return zip(field_names, types, strict=False)


def _parse_type_node(node, names=None) -> Any:
    """Resolve an AST node representing a type annotation into a runtime object.

    This function powers string-based signatures with explicit annotations.  It
    accepts AST nodes produced from fragments such as `int`, `list[str]`,
    `Optional[int]`, `MyModule.CustomType`, `int | None`, or
    `Field(...)`.

    Name resolution proceeds in this order:

    1. The supplied `names` mapping (seeded from `typing.__dict__` when
       `names` is `None`).
    2. A small set of common built-in types (`int`, `str`, `float`,
       `bool`, `list`, `tuple`, `dict`, `set`, `frozenset`,
       `complex`, `bytes`, `bytearray`).
    3. Dynamic `importlib.import_module` for the base name of dotted
       expressions such as `dspy.Image`.

    Args:
        node: AST node representing a type expression or supported annotation
            fragment.
        names: Optional mapping from annotation names to runtime objects.  If
            omitted, the function seeds the mapping from `typing.__dict__`
            and adds `NoneType`.

    Returns:
        Any: Runtime object represented by `node`.  In practice this is
        usually a Python type, a `typing` construct, a tuple of types, a
        constant value, or a Pydantic `Field` instance.

    Raises:
        ValueError: If the node contains an unknown name, an unsupported
            attribute, a malformed `Optional` type (wrong number of
            arguments), or an AST shape this parser does not handle.

    Examples:
        `int` resolves to `int`.

        `list[str]` resolves to `list[str]` (or the runtime equivalent).

        `Optional[int]` resolves to `typing.Optional[int]`.

        `int | None` (PEP 604) resolves to `typing.Optional[int]`.

        `dspy.Image` resolves by importing `dspy` and accessing `Image`.

        `Field(description="A field")` resolves to a Pydantic `Field`
        instance with the given keyword arguments.
    """

    if names is None:
        names = dict(typing.__dict__)
        names["NoneType"] = type(None)

    def resolve_name(type_name: str):
        # Check if it's a built-in known type or in the provided names
        if type_name in names:
            return names[type_name]
        # Common built-in types
        builtin_types = [int, str, float, bool, list, tuple, dict, set, frozenset, complex, bytes, bytearray]

        # Check if it matches any known built-in type by name
        for t in builtin_types:
            if t.__name__ == type_name:
                return t

        # Attempt to import a module with this name dynamically
        # This allows handling of module-based annotations like `dspy.Image`.
        try:
            mod = importlib.import_module(type_name)
            names[type_name] = mod
            return mod
        except ImportError:
            pass

        # If we don't know the type or module, raise an error
        raise ValueError(f"Unknown name: {type_name}")

    if isinstance(node, ast.Module):
        if len(node.body) != 1:
            raise ValueError(f"Code is not syntactically valid: {ast.dump(node)}")
        return _parse_type_node(node.body[0], names)

    if isinstance(node, ast.Expr):
        return _parse_type_node(node.value, names)

    if isinstance(node, ast.Name):
        return resolve_name(node.id)

    if isinstance(node, ast.Attribute):
        base = _parse_type_node(node.value, names)
        attr_name = node.attr

        if hasattr(base, attr_name):
            return getattr(base, attr_name)

        if isinstance(node.value, ast.Name):
            full_name = f"{node.value.id}.{attr_name}"
            if full_name in names:
                return names[full_name]

        raise ValueError(f"Unknown attribute: {attr_name} on {base}")

    if isinstance(node, ast.Subscript):
        base_type = _parse_type_node(node.value, names)
        slice_node = node.slice
        if isinstance(slice_node, ast.Index):  # For older Python versions
            slice_node = slice_node.value

        if isinstance(slice_node, ast.Tuple):
            arg_types = tuple(_parse_type_node(elt, names) for elt in slice_node.elts)
        else:
            arg_types = (_parse_type_node(slice_node, names),)

        # Special handling for Union, Optional
        if base_type is typing.Union:
            return typing.Union[arg_types]
        if base_type is typing.Optional:
            if len(arg_types) != 1:
                raise ValueError("Optional must have exactly one type argument")
            return typing.Optional[arg_types[0]]

        return base_type[arg_types]

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        # Handle PEP 604: int | None, str | float, etc.
        left = _parse_type_node(node.left, names)
        right = _parse_type_node(node.right, names)

        # Optional[X] is Union[X, NoneType]
        if right is type(None):
            return typing.Optional[left]
        if left is type(None):
            return typing.Optional[right]
        return typing.Union[left, right]

    if isinstance(node, ast.Tuple):
        return tuple(_parse_type_node(elt, names) for elt in node.elts)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Field":
        keys = [kw.arg for kw in node.keywords]
        values = []
        for kw in node.keywords:
            if isinstance(kw.value, ast.Constant):
                values.append(kw.value.value)
            else:
                values.append(_parse_type_node(kw.value, names))
        return Field(**dict(zip(keys, values, strict=False)))

    raise ValueError(
        f"Failed to parse string-base Signature due to unhandled AST node type in annotation: {ast.dump(node)}. "
        "Please consider using class-based DSPy Signatures instead."
    )


def infer_prefix(attribute_name: str) -> str:
    """Infer a human-readable prompt prefix from a Python attribute name.

    The transformation is designed for DSPy field labels. It inserts word
    boundaries for camelCase transitions, preserves acronym-style all-caps
    segments, separates adjacent letters and digits, and finally joins the
    words in title case.

    Warning:
        This function computes default `prefix` metadata for signature fields,
        but none of the adapters currently use field `prefix` values when
        building prompts. As a result, changing the inferred prefix may affect
        schema metadata but not what is shown to the language models.

    Args:
        attribute_name: Raw Python attribute name such as `camelCaseText` or
            `html_parser2`.

    Returns:
        str: Readable label suitable for use as a field prefix, without the
        trailing colon that callers typically add separately.

    Examples:
        >>> infer_prefix("camelCaseText")
        'Camel Case Text'
        >>> infer_prefix("snake_case_text")
        'Snake Case Text'
        >>> infer_prefix("text2number")
        'Text 2 Number'
        >>> infer_prefix("HTMLParser")
        'HTML Parser'
    """
    # Step 1: Convert camelCase to snake_case
    # Example: "camelCase" -> "camel_Case"
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", attribute_name)

    # Handle consecutive capitals
    # Example: "camel_Case" -> "camel_case"
    intermediate_name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)

    # Step 2: Handle numbers by adding underscores around them
    # Example: "text2number" -> "text_2_number"
    with_underscores_around_numbers = re.sub(
        r"([a-zA-Z])(\d)",  # Match letter followed by number
        r"\1_\2",  # Add underscore between them
        intermediate_name,
    )
    # Example: "2text" -> "2_text"
    with_underscores_around_numbers = re.sub(
        r"(\d)([a-zA-Z])",  # Match number followed by letter
        r"\1_\2",  # Add underscore between them
        with_underscores_around_numbers,
    )

    # Step 3: Convert to Title Case while preserving acronyms
    words = with_underscores_around_numbers.split("_")
    title_cased_words = []
    for word in words:
        if word.isupper():
            # Preserve acronyms like 'HTML', 'API' as-is
            title_cased_words.append(word)
        else:
            # Capitalize first letter: 'text' -> 'Text'
            title_cased_words.append(word.capitalize())

    # Join words with spaces
    # Example: ["Text", "2", "Number"] -> "Text 2 Number"
    return " ".join(title_cased_words)
