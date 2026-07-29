import base64
import binascii
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

    Construction and adapter parsing never access the filesystem; use :meth:`from_path` to read a
    local file, :meth:`from_bytes` for raw bytes, or :meth:`from_file_id` to reference a preuploaded
    file.

    Examples:
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

    @pydantic.field_validator("file_data")
    @classmethod
    def validate_file_data(cls, value: str | None) -> str | None:
        if value is None:
            return None

        header, separator, data = value.partition(",")
        media_type, *parameters = header.removeprefix("data:").split(";")
        if (
            not value.startswith("data:")
            or not separator
            or not media_type
            or "base64" not in {parameter.lower() for parameter in parameters}
        ):
            raise ValueError("file_data must have the form data:<mime_type>;base64,<base64_data>")
        try:
            base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("file_data must contain valid base64 data") from error
        return value

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
                return dict(values)
            raise ValueError("Value of `dspy.File` must contain at least one of: file_data, file_id, or filename")

        raise ValueError("File must be constructed from named fields or an explicit from_* factory")

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
                len_data = (
                    len(self.file_data.split("base64,")[1]) if "base64," in self.file_data else len(self.file_data)
                )
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
    def from_path(
        cls,
        path: str | os.PathLike[str],
        *,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> "File":
        """Create a File from a local file path.

        Args:
            path: Path to the file to read
            filename: Optional filename to use (defaults to basename of path)
            mime_type: Optional MIME type (defaults to auto-detection from file extension)
        """
        with open(path, "rb") as f:
            file_bytes = f.read()

        path_string = os.fspath(path)
        if filename is None:
            filename = os.path.basename(path_string)

        if mime_type is None:
            mime_type, _ = mimetypes.guess_type(path_string)
            if mime_type is None:
                mime_type = "application/octet-stream"

        encoded_data = base64.b64encode(file_bytes).decode("utf-8")
        file_data = f"data:{mime_type};base64,{encoded_data}"

        return cls(file_data=file_data, filename=filename)

    @classmethod
    def from_bytes(
        cls,
        file_bytes: bytes,
        *,
        filename: str | None = None,
        mime_type: str = "application/octet-stream",
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
    def from_file_id(cls, file_id: str, *, filename: str | None = None) -> "File":
        """Create a File from an uploaded file ID."""
        return cls(file_id=file_id, filename=filename)


def encode_file_to_dict(file_input: Any) -> dict:
    """
    Encode various file inputs to a dict with file_data, file_id, and/or filename.

    Args:
        file_input: Can be bytes, a data URI string, a structured dictionary, or a File instance.

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

    elif isinstance(file_input, dict):
        file_obj = File(**file_input)
        return encode_file_to_dict(file_obj)

    elif isinstance(file_input, str):
        if file_input.startswith("data:"):
            return {"file_data": File(file_data=file_input).file_data}
        raise ValueError(
            f"String file inputs must be data URIs, received: {file_input}. Load local files with File.from_path()."
        )

    elif isinstance(file_input, bytes):
        file_obj = File.from_bytes(file_input)
        return {"file_data": file_obj.file_data}

    else:
        raise ValueError(f"Unsupported file input type: {type(file_input)}")
