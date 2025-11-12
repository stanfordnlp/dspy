import base64
import mimetypes
import os
from typing import Any

import pydantic

from dspy.adapters.types.base_type import Type


class File(Type):
    """A file input type for DSPy.
    See https://platform.openai.com/docs/api-reference/chat/create#chat_create-messages-user_message-content-array_of_content_parts-file_content_part-file for specification.

    The file_data field should be a data URI with the format:
        data:<mime_type>;base64,<base64_encoded_data>

    Example:
        ```python
        import dspy

        class QA(dspy.Signature):
            file: dspy.File = dspy.InputField()
            summary = dspy.OutputField()
        program = dspy.Predict(QA)
        result = program(file=dspy.File.from_path("./research.pdf"))
        print(result.summary)
        ```
    """

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
            raise ValueError("Value of `dspy.File` must contain at least one of: file_data, file_id, or filename")

        return encode_file_to_dict(values)

    def format(self) -> list[dict[str, Any]]:
        try:
            file_dict = {}
            if self.file_data:
                file_dict["file_data"] = self.file_data
            if self.file_id:
                file_dict["file_id"] = self.file_id
            if self.filename:
                file_dict["filename"] = self.filename

            return [{"type": "file", "file": file_dict}]
        except Exception as e:
            raise ValueError(f"Failed to format file for DSPy: {e}")

    def __str__(self):
        return self.serialize_model()

    def __repr__(self):
        parts = []
        if self.file_data is not None:
            if self.file_data.startswith("data:"):
                # file data has "data:text/plain;base64,..." format
                mime_type = self.file_data.split(";")[0].split(":")[1]
                len_data = len(self.file_data.split("base64,")[1]) if "base64," in self.file_data else len(self.file_data)
                parts.append(f"file_data=<DATA_URI({mime_type}, {len_data} chars)>")
            else:
                len_data = len(self.file_data)
                parts.append(f"file_data=<DATA({len_data} chars)>")
        if self.file_id is not None:
            parts.append(f"file_id='{self.file_id}'")
        if self.filename is not None:
            parts.append(f"filename='{self.filename}'")
        return f"File({', '.join(parts)})"

    @classmethod
    def from_path(cls, file_path: str, filename: str | None = None, mime_type: str | None = None) -> "File":
        """Create a File from a local file path.

        Args:
            file_path: Path to the file to read
            filename: Optional filename to use (defaults to basename of path)
            mime_type: Optional MIME type (defaults to auto-detection from file extension)
        """
        if not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        if filename is None:
            filename = os.path.basename(file_path)

        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type is None:
                mime_type = "application/octet-stream"

        encoded_data = base64.b64encode(file_bytes).decode("utf-8")
        file_data = f"data:{mime_type};base64,{encoded_data}"

        return cls(file_data=file_data, filename=filename)

    @classmethod
    def from_bytes(
        cls, file_bytes: bytes, filename: str | None = None, mime_type: str = "application/octet-stream"
    ) -> "File":
        """Create a File from raw bytes.

        Args:
            file_bytes: Raw bytes of the file
            filename: Optional filename
            mime_type: MIME type (defaults to 'application/octet-stream')
        """
        encoded_data = base64.b64encode(file_bytes).decode("utf-8")
        file_data = f"data:{mime_type};base64,{encoded_data}"
        return cls(file_data=file_data, filename=filename)

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
