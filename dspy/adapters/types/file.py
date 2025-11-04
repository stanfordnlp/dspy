import base64
import os
from typing import Any

import pydantic

from dspy.adapters.types.base_type import Type


class File(Type):
    file_data: str | None = None
    file_id: str | None = None
    filename: str | None = None

    model_config = pydantic.ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, values: Any) -> Any:
        if isinstance(values, cls):
            return {
                "file_data": values.file_data,
                "file_id": values.file_id,
                "filename": values.filename,
            }

        if isinstance(values, dict):
            if "file_data" in values or "file_id" in values or "filename" in values:
                return values
            raise ValueError("Dict must contain at least one of: file_data, file_id, or filename")

        return encode_file_to_dict(values)

    def format(self) -> list[dict[str, Any]]:
        try:
            file_dict = {}
            if self.file_data is not None:
                file_dict["file_data"] = self.file_data
            if self.file_id is not None:
                file_dict["file_id"] = self.file_id
            if self.filename is not None:
                file_dict["filename"] = self.filename

            return [{"type": "file", "file": file_dict}]
        except Exception as e:
            raise ValueError(f"Failed to format file for DSPy: {e}")

    def __str__(self):
        return self.serialize_model()

    def __repr__(self):
        parts = []
        if self.file_data is not None:
            len_data = len(self.file_data)
            parts.append(f"file_data=<BASE64_ENCODED({len_data})>")
        if self.file_id is not None:
            parts.append(f"file_id='{self.file_id}'")
        if self.filename is not None:
            parts.append(f"filename='{self.filename}'")
        return f"File({', '.join(parts)})"

    @classmethod
    def from_path(cls, file_path: str, filename: str | None = None) -> "File":
        """Create a File from a local file path."""
        if not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            file_data = f.read()

        encoded_data = base64.b64encode(file_data).decode("utf-8")

        if filename is None:
            filename = os.path.basename(file_path)

        return cls(file_data=encoded_data, filename=filename)

    @classmethod
    def from_bytes(cls, file_bytes: bytes, filename: str | None = None) -> "File":
        """Create a File from raw bytes."""
        encoded_data = base64.b64encode(file_bytes).decode("utf-8")
        return cls(file_data=encoded_data, filename=filename)

    @classmethod
    def from_file_id(cls, file_id: str, filename: str | None = None) -> "File":
        """Create a File from an uploaded file ID."""
        return cls(file_id=file_id, filename=filename)


def encode_file_to_dict(file_input: Any) -> dict:
    """
    Encode various file inputs to a dict with file_data, file_id, and/or filename.

    Args:
        file_input: Can be a file path (str), bytes, or File instance.

    Returns:
        dict: A dictionary with file_data, file_id, and/or filename keys.
    """
    if isinstance(file_input, File):
        result = {}
        if file_input.file_data is not None:
            result["file_data"] = file_input.file_data
        if file_input.file_id is not None:
            result["file_id"] = file_input.file_id
        if file_input.filename is not None:
            result["filename"] = file_input.filename
        return result

    elif isinstance(file_input, str):
        if os.path.isfile(file_input):
            file_obj = File.from_path(file_input)
        else:
            raise ValueError(f"Unrecognized file string: {file_input}; must be a valid file path")

        return {
            "file_data": file_obj.file_data,
            "filename": file_obj.filename,
        }

    elif isinstance(file_input, bytes):
        file_obj = File.from_bytes(file_input)
        return {"file_data": file_obj.file_data}

    else:
        raise ValueError(f"Unsupported file input type: {type(file_input)}")


