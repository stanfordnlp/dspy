from typing import Any, Optional
from pydantic import BaseModel, Field


class Tool(BaseModel):
    tool: Any
    name: str
    desc: Optional[str]
    input_type: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.name}{f'(valid_input: {self.input_type})' if self.input_type else ''}: {self.desc}"

    def __repr__(self) -> str:
        return self.__str__()


class Action(BaseModel):
    tool_name: Any = Field(..., description="Name of the tool to use.")
    tool_input_query: Any = Field(..., description="Query to pass as input to the tool.")


class ActionOutput(BaseModel):
    tool_name: str
    tool_input_query: str
    tool_output: str