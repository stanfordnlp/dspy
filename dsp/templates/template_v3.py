from typing import Callable
from dsp.templates import TemplateV2, passages2text, format_answers, Field


class Type:
    """A primitive datatype that defines and represents a prompt label."""

    def __init__(self, prefix: str, desc: str, format=None) -> None:
        self.prefix = prefix
        self.desc = desc
        self.format = format

    def __call__(self, **kwargs):
        kwargs = {**self.__dict__, **kwargs}
        return Type(**kwargs)


class Template(TemplateV2):
    """A template datatype that represents the structure of communicate with the LM."""

    def __init__(self, instructions: str, **kwargs):
        self.instructions = instructions
        self.kwargs = kwargs

        self.fields: list[Field] = []
        self.format_handlers: dict[str, Callable] = {
            "context": passages2text,
            "answers": format_answers,
        }

        for key, value in kwargs.items():
            prefix: str = value.prefix
            separator: str = (
                " " if prefix.rstrip() == prefix and len(prefix) > 0 else prefix[len(prefix.rstrip()) :]
            )
            field = Field(
                name=prefix.strip(),
                description=value.desc,
                input_variable=key,
                output_variable=key,
                separator=separator,
            )
            self.fields.append(field)

            if value.format:
                self.format_handlers[key] = value.format
