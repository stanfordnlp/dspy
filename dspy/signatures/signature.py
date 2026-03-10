"""DSPy `Signature` class and string-based signature parsing.

A signature declares the input/output contract for a DSPy module.
Two syntaxes:

    # Class syntax - the docstring becomes the instruction text
    >>> import dspy
    >>> class Review(dspy.Signature):
    ...     \"\"\"
    ...     Summarize the review and give it a rating.
    ...     \"\"\"
    ...     # Inputs
    ...     text = dspy.InputField(desc="product review text")
    ...     # Outputs
    ...     summary = dspy.OutputField(desc="a one-sentence summary")
    ...     rating: int = dspy.OutputField(desc="rating", ge=0, le=5)
    >>> reviewer = dspy.Predict(Review)

    # String syntax
    >>> Review = dspy.Signature(
    ...     "text -> summary, rating: int",
    ...     instructions="Summarize the review and give it a rating."
    ... )

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
        """Detect custom types from the caller's frame based on the signature string.

        Note: This method relies on Python's frame introspection which has some limitations:
        1. May not work in all Python implementations (e.g., compiled with optimizations)
        2. Looks up a limited number of frames in the call stack
        3. Cannot find types that are imported but not in the caller's namespace

        For more reliable custom type resolution, explicitly provide types using the
        `custom_types` parameter when creating a Signature.
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
        # TODO: In a future major release, remove the notion of `prefix` from fields
        # entirely and update this logic accordingly.
        for name, field in cls.model_fields.items():
            if "prefix" not in field.json_schema_extra:
                field.json_schema_extra["prefix"] = infer_prefix(name) + ":"
            if "desc" not in field.json_schema_extra:
                field.json_schema_extra["desc"] = f"${{{name}}}"

        return cls

    def _validate_fields(cls):
        """Verify all dspy extra fields are marked with `InputField` or `OutputField`."""
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
    # Note: Don't put a docstring here, as it will become the default instructions
    # for any signature that doesn't define it's own instructions.

    @classmethod
    def with_instructions(cls, instructions: str) -> type["Signature"]:
        """Return a new signature with different instructions.

        The original signature is unchanged.  Fields, types, and
        metadata are copied as-is.  The returned class must be
        assigned to a predictor (or used to create one) to take effect.

        This is the primary mechanism DSPy optimizers use to swap in
        candidate instructions during prompt search.

        Args:
            instructions: Task description the language model will see.

        Returns:
            (type[Signature]): A new signature class.

        Examples:
            >>> import dspy
            >>> class QA(dspy.Signature):
            ...     "Answer the question."
            ...     question: str = dspy.InputField()
            ...     answer: str = dspy.OutputField()

            Assign the result to a predictor so the model sees it:

            >>> predict = dspy.Predict(QA)
            >>> predict.signature = predict.signature.with_instructions(
            ...     "Answer the question in French."
            ... )
            >>> predict.signature.instructions
            'Answer the question in French.'

            Chain with :meth:`with_updated_fields` for deeper edits:

            >>> updated = QA.with_instructions("Be concise.").with_updated_fields(
            ...     "answer", desc="A single-sentence answer"
            ... )

            Append to existing instructions instead of replacing:

            >>> enriched = QA.with_instructions(
            ...     QA.instructions + "\\n\\nAlways cite sources."
            ... )
            >>> enriched is not QA
            True
        """
        return Signature(cls.fields, instructions)

    @classmethod
    def with_updated_fields(cls, name: str, type_: type | None = None, **kwargs: dict[str, Any]) -> type["Signature"]:
        """Return a new signature with updated metadata on one field.

        The original signature is unchanged.  Only the named field is
        modified in the copy.  The returned class must be assigned to
        a predictor (or used to create one) to take effect.

        The two things you will typically update are `desc` (the
        field description the language model sees) and `type_` (the
        Python type annotation, e.g. `str`, `int`, `Literal[...]`).

        Args:
            name: Name of the field to update.
            type_: New Python type annotation (e.g. `int`,
                `list[str]`), or `None` to keep the current type.
            **kwargs: Field metadata updates.  The most common key
                is `desc` (the description the language model sees).

        Returns:
            (type[Signature]): A new signature class.

        Examples:
            >>> import dspy
            >>> class QA(dspy.Signature):
            ...     "Answer the question."
            ...     question: str = dspy.InputField(desc="A factual question")
            ...     answer: str = dspy.OutputField(desc="Short answer")

            Update a field description and assign back to a predictor:

            >>> predict = dspy.Predict(QA)
            >>> predict.signature = predict.signature.with_updated_fields(
            ...     "answer", desc="A single-sentence answer with citation"
            ... )
            >>> predict.signature.fields["answer"].json_schema_extra["desc"]
            'A single-sentence answer with citation'

            Change a field's type annotation:

            >>> from typing import Literal
            >>> Action = Literal["search", "lookup", "finish"]
            >>> sig2 = QA.with_updated_fields("answer", Action)
            >>> sig2.fields["answer"].annotation is Action
            True
            >>> sig2 is not QA
            True
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
        """Return a new signature with a field added at the start of its section.

        The field goes into inputs or outputs depending on whether
        `field` is an `InputField` or `OutputField`.  The original
        signature is unchanged.  The returned class must be assigned
        to a predictor (or used to create one) to take effect.

        Args:
            name: Name for the new field.
            field: An `InputField()` or `OutputField()` instance.
            type_: Python type annotation (default `str`).

        Returns:
            (type[Signature]): A new signature class.

        Examples:
            Add a `reasoning` field before the existing outputs:

            >>> import dspy
            >>> class QA(dspy.Signature):
            ...     question: str = dspy.InputField()
            ...     answer: str = dspy.OutputField()
            >>> predict = dspy.Predict(
            ...     QA.prepend("reasoning", dspy.OutputField(desc="Think step by step"))
            ... )
            >>> list(predict.signature.output_fields.keys())
            ['reasoning', 'answer']

            Add `context` before the existing inputs:

            >>> with_context = QA.prepend(
            ...     "context", dspy.InputField(desc="Supporting passages")
            ... )
            >>> list(with_context.input_fields.keys())
            ['context', 'question']
        """
        return cls.insert(0, name, field, type_)

    @classmethod
    def append(cls, name, field, type_=None) -> type["Signature"]:
        """Return a new signature with a field added at the end of its section.

        The field goes into inputs or outputs depending on whether
        `field` is an `InputField` or `OutputField`.  The original
        signature is unchanged.  The returned class must be assigned
        to a predictor (or used to create one) to take effect.

        Args:
            name: Name for the new field.
            field: An `InputField()` or `OutputField()` instance.
            type_: Python type annotation (default `str`).

        Returns:
            (type[Signature]): A new signature class.

        Examples:
            Add a confidence score after the existing outputs:

            >>> import dspy
            >>> class QA(dspy.Signature):
            ...     question: str = dspy.InputField()
            ...     answer: str = dspy.OutputField()
            >>> extended = QA.append(
            ...     "confidence", dspy.OutputField(desc="0.0-1.0"), float
            ... )
            >>> list(extended.output_fields.keys())
            ['answer', 'confidence']

            Inject a hint as an extra input (used by refinement loops):

            >>> with_hint = QA.append(
            ...     "hint", dspy.InputField(desc="Advice from a previous run")
            ... )
            >>> list(with_hint.input_fields.keys())
            ['question', 'hint']
        """
        return cls.insert(-1, name, field, type_)

    @classmethod
    def delete(cls, name) -> type["Signature"]:
        """Return a new signature without the named field.

        The original signature is unchanged.  The returned class must
        be assigned to a predictor (or used to create one) to take
        effect.  If the field does not exist the call is a no-op
        (no error is raised).

        Args:
            name: Field name to remove.

        Returns:
            (type[Signature]): A new signature class.

        Examples:
            Remove a field that is handled natively by the LM:

            >>> import dspy
            >>> class QA(dspy.Signature):
            ...     question: str = dspy.InputField()
            ...     reasoning: str = dspy.OutputField()
            ...     answer: str = dspy.OutputField()
            >>> shorter = QA.delete("reasoning")
            >>> list(shorter.output_fields.keys())
            ['answer']

            Missing names are silently ignored:

            >>> shorter.delete("nonexistent") is not shorter
            True
        """
        fields = dict(cls.fields)

        fields.pop(name, None)

        return Signature(fields, cls.instructions)

    @classmethod
    def insert(cls, index: int, name: str, field, type_: type | None = None) -> type["Signature"]:
        """Return a new signature with a field at a specific position.

        The field goes into inputs or outputs depending on whether
        `field` is an `InputField` or `OutputField`.  The original
        signature is unchanged.  The returned class must be assigned
        to a predictor (or used to create one) to take effect.

        Most callers should prefer `prepend` (index 0) or `append`
        (index -1); use `insert` when you need a field between two
        existing ones.

        Args:
            index: Position within the section.  Use `0` for the
                start, `-1` for the end.
            name: Name for the new field.
            field: An `InputField()` or `OutputField()` instance.
            type_: Python type annotation (default `str`).

        Returns:
            (type[Signature]): A new signature class.

        Raises:
            ValueError: If `index` is out of range for the section.

        Examples:
            Insert a context field at the start of inputs:

            >>> import dspy
            >>> class QA(dspy.Signature):
            ...     question: str = dspy.InputField()
            ...     answer: str = dspy.OutputField()
            >>> extended = QA.insert(
            ...     0, "context", dspy.InputField(desc="Supporting passages")
            ... )
            >>> list(extended.input_fields.keys())
            ['context', 'question']

            Append with `-1`:

            >>> extended2 = QA.insert(
            ...     -1, "confidence", dspy.OutputField(desc="0.0-1.0"), float
            ... )
            >>> list(extended2.output_fields.keys())
            ['answer', 'confidence']
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
        """Test whether two signatures have the same instructions and fields.

        Compares instructions and per-field metadata (`desc`,
        `prefix`, …).  Does not compare type annotations, defaults,
        or validators.

        Args:
            other: Another signature class to compare against.

        Returns:
            (bool): `True` if instructions and field metadata match.

        Examples:
            >>> import dspy
            >>> class QA(dspy.Signature):
            ...     question: str = dspy.InputField()
            ...     answer: str = dspy.OutputField()

            A copy with the same metadata is equal:

            >>> QA.equals(QA.with_instructions(QA.instructions))
            True

            Different instructions make them unequal:

            >>> QA.equals(QA.with_instructions("Be concise."))
            False
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
        """Serialize the signature's mutable prompt state to a dict.

        Captures instructions and per-field `desc` — everything an
        optimizer might change.  Field names, types, and validators
        are not included; those are fixed by the class definition.
        Restore with `load_state`.

        Returns:
            (dict): A dict with `"instructions"` and `"fields"` keys.

        Examples:
            >>> import dspy
            >>> class QA(dspy.Signature):
            ...     "Answer the question."
            ...     question: str = dspy.InputField()
            ...     answer: str = dspy.OutputField()
            >>> state = QA.dump_state()
            >>> state["instructions"]
            'Answer the question.'
            >>> state["fields"][0]["description"]
            '${question}'
        """
        state = {"instructions": cls.instructions, "fields": []}
        for field in cls.fields:
            state["fields"].append(
                {
                    # TODO: In a future major release, stop serializing `prefix`
                    # (see matching TODO in field.py and load_state below).
                    "prefix": cls.fields[field].json_schema_extra["prefix"],
                    "description": cls.fields[field].json_schema_extra["desc"],
                }
            )

        return state

    @classmethod
    def load_state(cls, state):
        """Return a new signature with state restored from `dump_state`.

        The original signature is unchanged.  Fields in `state` are
        matched positionally against the current field order, not by
        name.

        Args:
            state: Dict previously produced by `dump_state`.

        Returns:
            (type[Signature]): A new signature class.

        Examples:
            Round-trip through `dump_state` / `load_state`:

            >>> import dspy
            >>> class QA(dspy.Signature):
            ...     "Answer the question."
            ...     question: str = dspy.InputField()
            ...     answer: str = dspy.OutputField()
            >>> restored = QA.load_state(QA.dump_state())
            >>> restored.equals(QA)
            True
            >>> restored is not QA
            True
        """
        signature_copy = Signature(deepcopy(cls.fields), cls.instructions)

        signature_copy.instructions = state["instructions"]
        for field, saved_field in zip(signature_copy.fields.values(), state["fields"], strict=False):
            # TODO: In a future major release, stop restoring `prefix`
            # (see matching TODO in field.py and dump_state above).
            field.json_schema_extra["prefix"] = saved_field["prefix"]
            field.json_schema_extra["desc"] = saved_field["description"]

        return signature_copy


def ensure_signature(signature: str | type[Signature], instructions=None) -> type[Signature]:
    """Coerce a string or signature class into a signature class.

    Every built-in DSPy module (`Predict`, `ChainOfThought`, …)
    calls this in its `__init__` so that users can pass either a
    shorthand string or a full class.  `None` is passed through
    unchanged.

    Args:
        signature: `"question -> answer"`, a `Signature` subclass,
            or `None`.
        instructions: Optional instruction text (only allowed when
            `signature` is a string).

    Returns:
        (type[Signature] | None): A signature class, or `None` if
            `None` was passed.

    Raises:
        ValueError: If `instructions` is given with a class signature.

    Examples:
        Strings are converted to a signature class:

        >>> from dspy.signatures.signature import ensure_signature
        >>> import dspy
        >>> sig = ensure_signature("question -> answer")
        >>> list(sig.fields.keys())
        ['question', 'answer']

        Existing classes pass through unchanged:

        >>> class QA(dspy.Signature):
        ...     question: str = dspy.InputField()
        ...     answer: str = dspy.OutputField()
        >>> ensure_signature(QA) is QA
        True

        Add instructions to a string shorthand:

        >>> sig2 = ensure_signature("query -> result", "Find the best match.")
        >>> sig2.instructions
        'Find the best match.'
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
    """Create a `Signature` subclass from a string or field dict.

    This is the constructor behind `dspy.Signature("...")`.  Most
    users should prefer the class syntax or the string shorthand;
    call `make_signature` directly when you need to build signatures
    programmatically from a dict of fields or when you need to set
    `signature_name` or `custom_types`.

    Args:
        signature: `"input1, input2 -> output1"` or a dict of
            `{name: (type, FieldInfo)}` pairs.
        instructions: Task description.  If omitted a default is
            synthesized from the field names.
        signature_name: Class name for the generated subclass.
        custom_types: Name-to-type mapping for resolving annotations
            in string signatures (e.g. `{"Passage": Passage}`).

    Returns:
        (type[Signature]): A new signature subclass.

    Examples:
        From a string shorthand:

        >>> sig1 = make_signature("question, context -> answer")
        >>> sig1.signature
        'question, context -> answer'

        From a dict (useful when building signatures programmatically):

        >>> sig2 = make_signature(
        ...     {"question": (str, InputField(desc="A factual question")),
        ...      "answer": (str, OutputField(desc="Short answer"))},
        ...     instructions="Answer the question.",
        ... )
        >>> sig2.instructions
        'Answer the question.'
        >>> list(sig2.fields.keys())
        ['question', 'answer']

        Use `custom_types` when a string signature references a type
        the parser cannot auto-resolve:

        >>> from pydantic import BaseModel
        >>> class Passage(BaseModel):
        ...     text: str
        ...     source: str
        >>> sig3 = make_signature(
        ...     "question, passages: list[Passage] -> answer",
        ...     instructions="Answer using the passages.",
        ...     custom_types={"Passage": Passage},
        ... )
        >>> list(sig3.fields.keys())
        ['question', 'passages', 'answer']

        Without `custom_types` the parser raises a `ValueError`:

        >>> make_signature("q: Passage -> a", instructions="x")
        Traceback (most recent call last):
            ...
        ValueError: Unknown name: Passage
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
        # Build a temporary signature from the already-parsed fields so we
        # don't re-enter make_signature (which would lose custom_types).
        tmp = create_model("_Tmp", __base__=Signature, __doc__="", **fixed_fields)
        instructions = _default_instructions(tmp)

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
    """Infer a human-readable label from a Python attribute name.

    Converts `snake_case`, `camelCase`, and mixed names into
    title-cased words (e.g. `"snake_case_text"` → `"Snake Case Text"`).

    .. deprecated::
        No current adapter uses field `prefix` values when building
        prompts.  This function will be removed in a future major
        release (see matching TODOs in `field.py`, `dump_state`, and
        `load_state`).

    Args:
        attribute_name: Python attribute name (e.g. `"camelCaseText"`).

    Returns:
        (str): Title-cased label without a trailing colon.

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
