import re
from typing import Any, ClassVar

import pydantic
from pydantic import create_model

from dspy.adapters.types.base_type import Type


class Code(Type):
    """Code type for code generation and analysis in DSPy.

    Supports optional language specification via ``dspy.Code["python"]`` syntax.
    When no language is specified, the model chooses the appropriate language.
    """

    model_config = pydantic.ConfigDict(
        json_schema_extra={"description": "Code object with a single `code` string field."},
    )

    code: str

    language: ClassVar[str] = ""

    def format(self):
        return f"{self.code}"

    @pydantic.model_serializer()
    def serialize_model(self):
        """Override to bypass the <<CUSTOM-TYPE-START-IDENTIFIER>> and <<CUSTOM-TYPE-END-IDENTIFIER>> tags."""
        return self.format()

    @classmethod
    def description(cls) -> str:
        base = (
            "Code represented in a string, specified in the `code` field. If this is an output field, the code "
            "field should follow the markdown code block format, e.g. "
        )
        if cls.language:
            base += f"\n```{cls.language.lower()}\n{{code}}\n```"
            base += f"\nProgramming language: {cls.language}"
        else:
            base += "\n```\n{code}\n```"
        return base

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls):
            return data

        if isinstance(data, str):
            return {"code": _filter_code(data)}

        if isinstance(data, dict):
            if "code" not in data:
                raise ValueError("`code` field is required for `dspy.Code`")
            if not isinstance(data["code"], str):
                raise ValueError(f"`code` field must be a string, but received type: {type(data['code'])}")
            return {"code": _filter_code(data["code"])}

        raise ValueError(f"Received invalid value for `dspy.Code`: {data}")


def _filter_code(code: str) -> str:
    """Extract code from markdown code blocks, stripping any language identifier."""
    # Case 1: format like:
    # ```python
    # {code_block}
    # ```
    regex_pattern = r"```(?:[^\n]*)\n(.*?)```"
    match = re.search(regex_pattern, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Case 2: ```<code>``` (no language, single-line)
    regex_pattern_simple = r"```(.*?)```"
    match = re.search(regex_pattern_simple, code, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback case
    return code


# Patch __class_getitem__ directly on the class to support dspy.Code["python"] syntax
def _code_class_getitem(cls, language):
    code_with_language_cls = create_model(f"{cls.__name__}_{language}", __base__=cls)
    code_with_language_cls.language = language
    return code_with_language_cls


Code.__class_getitem__ = classmethod(_code_class_getitem)
