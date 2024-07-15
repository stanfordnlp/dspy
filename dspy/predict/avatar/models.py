from typing import Any, Optional
from pydantic import BaseModel, Field


class Tool(BaseModel):
    tool: Any
    name: str
    desc: Optional[str]

    def __str__(self) -> str:
        return f"{self.name}: {self.desc}"

    def __repr__(self) -> str:
        return self.__str__()


class Action(BaseModel):
    tool_name: Any = Field(..., description="Name of the tool to use.")
    tool_input_query: Any = Field(..., description="Query to pass as input to the tool.")
