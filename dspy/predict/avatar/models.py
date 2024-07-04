from pydantic import BaseModel
from typing import Any, Optional


class Tool(BaseModel):
    tool: Any
    name: str
    tool_input_query: str
    desc: Optional[str]

    def __str__(self) -> str:
        return f"{self.name}({self.input_variable}): {self.desc}"

    def __repr__(self) -> str:
        return self.__str__()


class Action(BaseModel):
    rationale: str
    tool_name: str
    tool_input_query: str