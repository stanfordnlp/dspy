from collections import namedtuple
from typing import Callable

from .utils import format_answers, passages2text

Field = namedtuple("Field", "name separator input_variable output_variable description")


class Type:
    """A primitive datatype that defines and represents a prompt label."""

    def __init__(self, prefix: str, desc: str, format=None) -> None:
        self.prefix = prefix
        self.desc = desc
        self.format = format

    def __call__(self, **kwargs):
        kwargs = {**self.__dict__, **kwargs}
        return Type(**kwargs)

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Type) and self.__dict__ == __value.__dict__


class BaseTemplate:
    """A template datatype that represents the structure of communicate with the LM."""

    def __init__(self, instructions: str, **kwargs):
        self.instructions = instructions
        self.kwargs = kwargs

        self.fields: list[Field] = []
        self.format_handlers: dict[str, Callable] = {
            "context": passages2text,
            "passages": passages2text,
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
        
    
    # equality
    def __eq__(self, other):
        if set(self.kwargs.keys()) != set(other.kwargs.keys()):
            print('here2')
            return False

        for k in self.kwargs.keys():
            v1, v2 = self.kwargs[k], other.kwargs[k]
            if not v1 == v2:
                print(k, v1, v2)
            
        return self.instructions == other.instructions and self.kwargs == other.kwargs

    def __str__(self) -> str:
        # field names
        field_names = [field.name for field in self.fields]

        return f"Template({self.instructions}, {field_names})"
    
