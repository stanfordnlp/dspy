"""Opt-in text components for top-level signature field descriptions.

Component IDs are opaque JSON tuples, not dotted paths. Predictor names retain
GEPA's legacy IDs. No field names, types, nested models, or tools are optimized.
"""

import json
from copy import deepcopy
from dataclasses import dataclass

from pydantic import create_model


@dataclass(frozen=True)
class FieldComponent:
    predictor: str
    field: str


def field_components(program, reserved_keys=()):
    components = {}
    reserved = set(reserved_keys) | {name for name, _ in program.named_predictors()}
    for name, pred in program.named_predictors():
        for field in pred.signature.fields:
            key = "@field:" + json.dumps([name, field], separators=(",", ":"))
            if key in reserved:
                raise ValueError(f"Field component ID collides with a program component: {key}")
            components[key] = FieldComponent(name, field)
    return components


def field_values(program, components):
    predictors = dict(program.named_predictors())
    return {
        key: predictors[c.predictor].signature.fields[c.field].json_schema_extra["desc"]
        for key, c in components.items()
    }


def update_signature(signature, instructions, descriptions):
    """Copy fields onto a derived signature, retaining validators and defaults."""
    fields = {}
    for name, field in signature.fields.items():
        field = deepcopy(field)
        if name in descriptions:
            field.json_schema_extra = {**field.json_schema_extra, "desc": descriptions[name]}
        fields[name] = (field.annotation, field)
    return create_model(signature.__name__, __base__=signature, __doc__=instructions, **fields)


def metadata_contract(value):
    """A process-stable structural guard, not a fingerprint of executable code."""
    from dataclasses import fields, is_dataclass

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [metadata_contract(item) for item in value]
    if isinstance(value, dict):
        return {key: metadata_contract(item) for key, item in value.items()}
    if callable(value):
        return {
            "callable": (
                getattr(value, "__module__", type(value).__module__),
                getattr(value, "__qualname__", type(value).__qualname__),
            )
        }
    if is_dataclass(value):
        return {
            "type": (type(value).__module__, type(value).__qualname__),
            "fields": {field.name: metadata_contract(getattr(value, field.name)) for field in fields(value)},
        }
    return {"type": (type(value).__module__, type(value).__qualname__)}


def annotation_contract(annotation):
    """Normalize generic arguments too, including nested Annotated validators."""
    from typing import get_args, get_origin

    origin = get_origin(annotation)
    if origin is not None:
        return {"origin": metadata_contract(origin), "args": [annotation_contract(arg) for arg in get_args(annotation)]}
    return metadata_contract(annotation)
