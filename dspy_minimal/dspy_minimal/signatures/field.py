from pydantic import Field


def InputField(desc=None, **kwargs):
    """Create an input field for a signature."""
    extra = kwargs.get("json_schema_extra", {})
    extra["__dspy_field_type"] = "input"
    extra["desc"] = desc
    kwargs["json_schema_extra"] = extra
    return Field(**kwargs)


def OutputField(desc=None, **kwargs):
    """Create an output field for a signature."""
    extra = kwargs.get("json_schema_extra", {})
    extra["__dspy_field_type"] = "output"
    extra["desc"] = desc
    kwargs["json_schema_extra"] = extra
    return Field(**kwargs) 