"""DSPy `Signature` class and string-based signature parsing.

A signature declares the input/output contract for a DSPy module.
Two syntaxes::

    # Class syntax — the docstring becomes the instruction text
    class QA(dspy.Signature):
        '''Answer the question.'''
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    # String syntax
    QA = dspy.Signature("question -> answer")

All field-manipulation methods (`append`, `prepend`, `insert`, `delete`,
`with_instructions`, `with_updated_fields`) return a **new** signature
class; the original is never mutated.

Note: the base `Signature` class cannot carry a class docstring because
subclasses would inherit it as their default instructions.
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
    """Metaclass for DSPy signatures.

    `Signature("x -> y")` returns a new subclass (not an instance).
    Subclass creation validates fields, fills in defaults, and
    preserves declaration order.
    """

    def __call__(cls, *args, **kwargs):
        """Dispatch: `Signature(str)` builds a new subclass; `MySig(...)` creates an instance.

        When called on the base `Signature` class with a string,
        delegates to `make_signature`. User-defined annotation names
        are auto-resolved from the caller's stack frame when
        `custom_types` is not provided.
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
        """Resolve user-defined type names in a string signature via stack introspection.

        Only top-level annotation names (after `:`) are detected.
        Names nested in generics (e.g. `Passage` in `list[Passage]`)
        need explicit `custom_types`.
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
        """Build a new signature class.

        Fills in default annotations (`str`), preserves field order,
        synthesizes instructions if none given, validates that every
        field uses `InputField`/`OutputField`, and sets `prefix`/`desc`
        defaults.
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
        """Verify all fields are marked with `InputField` or `OutputField`."""
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
        """The signature's instruction text (from the class docstring)."""
        return inspect.cleandoc(getattr(cls, "__doc__", ""))

    @instructions.setter
    def instructions(cls, instructions: str) -> None:
        """Set new instruction text."""
        cls.__doc__ = instructions

    @property
    def input_fields(cls) -> dict[str, FieldInfo]:
        """Input fields in declaration order."""
        return cls._get_fields_with_type("input")

    @property
    def output_fields(cls) -> dict[str, FieldInfo]:
        """Output fields in declaration order."""
        return cls._get_fields_with_type("output")

    @property
    def fields(cls) -> dict[str, FieldInfo]:
        """All fields, inputs first then outputs."""
        # Make sure to give input fields before output fields
        return {**cls.input_fields, **cls.output_fields}

    @property
    def signature(cls) -> str:
        """Compact string form: `"input1, input2 -> output1, output2"`."""
        input_fields = ", ".join(cls.input_fields.keys())
        output_fields = ", ".join(cls.output_fields.keys())
        return f"{input_fields} -> {output_fields}"

    def _get_fields_with_type(cls, field_type) -> dict[str, FieldInfo]:
        """Filter fields by DSPy type tag (`"input"` or `"output"`)."""
        return {k: v for k, v in cls.model_fields.items() if v.json_schema_extra["__dspy_field_type"] == field_type}

    def __repr__(cls):
        """Debug representation with fields and instructions."""
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
        """Return a new signature with different instructions.

        The original signature is unchanged. Fields, types, and
        metadata are copied as-is.

        Args:
            instructions: New instruction text for the returned
                signature.

        Returns:
            type[Signature]: A new signature class.

        Examples:
            >>> import dspy
            >>> class MySig(dspy.Signature):
            ...     input_text: str = dspy.InputField(desc="Input text")
            ...     output_text: str = dspy.OutputField(desc="Output text")
            >>> NewSig = MySig.with_instructions("Translate to French.")
            >>> NewSig.instructions
            'Translate to French.'
        """
        return Signature(cls.fields, instructions)

    @classmethod
    def with_updated_fields(cls, name: str, type_: type | None = None, **kwargs: dict[str, Any]) -> type["Signature"]:
        """Return a new signature with updated metadata on one field.

        The original signature is unchanged. Only the named field is
        modified in the copy.

        Args:
            name: Field to update.
            type_: Optional new type annotation.
            **kwargs: Metadata entries merged into the field's
                `json_schema_extra` (e.g. `desc`, `prefix`).

        Returns:
            type[Signature]: A new signature class.
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
        """Add a field before all others in its section (inputs or outputs).

        Whether the field is placed among inputs or outputs is
        determined by the field itself (i.e. `InputField` vs
        `OutputField`). The original signature is unchanged.

        Args:
            name: Name for the new field.
            field: An `InputField()` or `OutputField()` instance.
            type_: Optional type annotation (defaults to `str`).

        Returns:
            type[Signature]: A new signature class.

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
        """Add a field after all others in its section (inputs or outputs).

        Whether the field is placed among inputs or outputs is
        determined by the field itself (i.e. `InputField` vs
        `OutputField`). The original signature is unchanged.

        Args:
            name: Name for the new field.
            field: An `InputField()` or `OutputField()` instance.
            type_: Optional type annotation (defaults to `str`).

        Returns:
            type[Signature]: A new signature class.

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
        """Return a new signature without the named field.

        If the field doesn't exist, returns an equivalent copy without
        raising an error. The original signature is unchanged.

        Args:
            name: Field name to remove.

        Returns:
            type[Signature]: A new signature class.

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
        """Insert a field at a specific position within its section.

        Input fields are inserted among inputs, output fields among
        outputs. Negative indices are supported (`-1` appends).
        The original signature is unchanged.

        Args:
            index: Position within the section. Negative values count
                from the end; `-1` inserts at the end.
            name: Name for the new field.
            field: An `InputField()` or `OutputField()` instance.
            type_: Optional type annotation (defaults to `str`).

        Returns:
            type[Signature]: A new signature class.

        Raises:
            ValueError: If `index` is out of range for the section.

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
        """Test whether two signatures have the same instructions and field metadata.

        Compares instructions and `json_schema_extra` for each field.
        Does not compare type annotations, defaults, or validators.
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
        """Serialize the signature's mutable state to a dict.

        Captures instructions and per-field `prefix`/`desc` metadata.
        Field names, types, and validators are not included — those
        are fixed by the class definition.

        Returns:
            dict: `{"instructions": str, "fields": [{"prefix": ..., "description": ...}, ...]}`.
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
        """Create a new signature with state restored from `dump_state`.

        The original signature is unchanged. Fields in `state` are
        matched positionally against the current field order, not by
        name.

        Args:
            state: Dict previously produced by `dump_state`.

        Returns:
            type[Signature]: A new signature class with restored
                instructions and field labels.
        """
        signature_copy = Signature(deepcopy(cls.fields), cls.instructions)

        signature_copy.instructions = state["instructions"]
        for field, saved_field in zip(signature_copy.fields.values(), state["fields"], strict=False):
            field.json_schema_extra["prefix"] = saved_field["prefix"]
            field.json_schema_extra["desc"] = saved_field["description"]

        return signature_copy


def ensure_signature(signature: str | type[Signature], instructions=None) -> type[Signature]:
    """Coerce a string, signature class, or `None` into a signature class.

    Useful at API boundaries where callers may pass either a string
    like `"question -> answer"` or an existing signature class.
    `None` is passed through unchanged.

    Args:
        signature: A string signature, a `Signature` subclass, or `None`.
        instructions: Optional instruction text (only valid when
            `signature` is a string).

    Returns:
        type[Signature] | None: A signature class, or `None` if `None`
            was passed.

    Raises:
        ValueError: If `instructions` is given with a non-string signature.
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
    """Create a `Signature` subclass from a string or field mapping.

    This is the constructor behind `Signature("...")`. Most users
    won't call it directly — use the class syntax or
    `dspy.Signature("question -> answer")` instead. It's useful when
    you need to set `signature_name` or supply `custom_types` for
    annotations the parser can't auto-resolve.

    Args:
        signature: `"input1, input2 -> output1"` or a dict of
            `{name: (type, FieldInfo)}` pairs.
        instructions: Instruction text. Synthesized from field names
            if omitted.
        signature_name: Class name for the generated subclass.
        custom_types: Name-to-type mapping for resolving annotations
            in string signatures (e.g. `{"Passage": Passage}`).

    Returns:
        type[Signature]: A new signature subclass.

    Examples:
        >>> sig1 = make_signature("question, context -> answer")
        >>> sig1.signature
        'question, context -> answer'

        >>> sig2 = make_signature({
        ...     "question": (str, InputField(desc="Question to answer")),
        ...     "answer": (str, OutputField(desc="Final answer")),
        ... })
        >>> list(sig2.fields.keys())
        ['question', 'answer']
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
    """Split a `"inputs -> outputs"` string into `{name: (type, FieldInfo)}` pairs.

    The string must contain exactly one `->`. Fields on the left become
    `InputField` definitions, fields on the right become `OutputField`
    definitions.
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
    """Parse one side of a `->` split into `(field_name, type)` pairs.

    Uses Python's own parser by embedding the string as a function
    parameter list. Fields without annotations default to `str`.
    """

    args = ast.parse(f"def f({field_string}): pass").body[0].args.args
    field_names = [arg.arg for arg in args]
    types = [str if arg.annotation is None else _parse_type_node(arg.annotation, names) for arg in args]
    return zip(field_names, types, strict=False)


def _parse_type_node(node, names=None) -> Any:
    """Resolve an AST type-annotation node into a runtime Python object.

    Handles `int`, `list[str]`, `Optional[int]`, `int | None`,
    `dspy.Image`, `Field(...)`, etc. Falls back to `names` mapping,
    built-in types, then `importlib.import_module`.
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
