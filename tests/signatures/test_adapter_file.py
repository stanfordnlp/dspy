import os
import tempfile

import pydantic
import pytest

import dspy
from dspy.adapters.types.file import encode_file_to_dict
from dspy.utils.dummies import DummyLM


@pytest.fixture
def sample_text_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp_file:
        tmp_file.write("This is a test file.")
        tmp_file_path = tmp_file.name
    yield tmp_file_path
    try:
        os.unlink(tmp_file_path)
    except Exception:
        pass


def count_messages_with_file_pattern(messages):
    pattern = {"type": "file", "file": lambda x: isinstance(x, dict)}

    def check_pattern(obj, pattern):
        if isinstance(pattern, dict):
            if not isinstance(obj, dict):
                return False
            return all(k in obj and check_pattern(obj[k], v) for k, v in pattern.items())
        if callable(pattern):
            return pattern(obj)
        return obj == pattern

    def count_patterns(obj, pattern):
        count = 0
        if check_pattern(obj, pattern):
            count += 1
        if isinstance(obj, dict):
            count += sum(count_patterns(v, pattern) for v in obj.values())
        if isinstance(obj, list | tuple):
            count += sum(count_patterns(v, pattern) for v in obj)
        return count

    return count_patterns(messages, pattern)


def setup_predictor(signature, expected_output):
    lm = DummyLM([expected_output])
    dspy.settings.configure(lm=lm)
    return dspy.Predict(signature), lm


def test_file_from_local_path(sample_text_file):
    file_obj = dspy.File.from_path(sample_text_file)
    assert file_obj.file_data is not None
    assert file_obj.file_data.startswith("data:text/plain;base64,")
    assert file_obj.filename == os.path.basename(sample_text_file)


def test_file_from_path_method(sample_text_file):
    file_obj = dspy.File.from_path(sample_text_file)
    assert file_obj.file_data is not None
    assert file_obj.file_data.startswith("data:text/plain;base64,")
    assert file_obj.filename == os.path.basename(sample_text_file)


def test_file_from_path_with_custom_filename(sample_text_file):
    file_obj = dspy.File.from_path(sample_text_file, filename="custom.txt")
    assert file_obj.file_data is not None
    assert file_obj.file_data.startswith("data:text/plain;base64,")
    assert file_obj.filename == "custom.txt"


def test_file_from_bytes():
    file_bytes = b"Test file content"
    file_obj = dspy.File.from_bytes(file_bytes)
    assert file_obj.file_data is not None
    assert file_obj.file_data.startswith("data:application/octet-stream;base64,")
    assert file_obj.filename is None


def test_file_from_bytes_with_filename():
    file_bytes = b"Test file content"
    file_obj = dspy.File.from_bytes(file_bytes, filename="test.txt")
    assert file_obj.file_data is not None
    assert file_obj.file_data.startswith("data:application/octet-stream;base64,")
    assert file_obj.filename == "test.txt"


def test_file_from_file_id():
    file_obj = dspy.File.from_file_id("file-abc123")
    assert file_obj.file_id == "file-abc123"
    assert file_obj.file_data is None


def test_file_from_file_id_with_filename():
    file_obj = dspy.File.from_file_id("file-abc123", filename="document.pdf")
    assert file_obj.file_id == "file-abc123"
    assert file_obj.filename == "document.pdf"


def test_file_from_dict_with_file_data():
    file_obj = dspy.File(file_data="data:text/plain;base64,dGVzdA==", filename="test.txt")
    assert file_obj.file_data == "data:text/plain;base64,dGVzdA=="
    assert file_obj.filename == "test.txt"


def test_file_from_dict_with_file_id():
    file_obj = dspy.File(file_id="file-xyz789")
    assert file_obj.file_id == "file-xyz789"


def test_file_format_with_file_data():
    file_obj = dspy.File.from_bytes(b"test", filename="test.txt")
    formatted = file_obj.format()
    assert isinstance(formatted, list)
    assert len(formatted) == 1
    assert formatted[0]["type"] == "file"
    assert "file" in formatted[0]
    assert "file_data" in formatted[0]["file"]
    assert "filename" in formatted[0]["file"]


def test_file_format_with_file_id():
    file_obj = dspy.File.from_file_id("file-123")
    formatted = file_obj.format()
    assert formatted[0]["type"] == "file"
    assert formatted[0]["file"]["file_id"] == "file-123"


def test_file_repr_with_file_data():
    file_obj = dspy.File.from_bytes(b"Test content", filename="test.txt")
    repr_str = repr(file_obj)
    assert "DATA_URI" in repr_str
    assert "application/octet-stream" in repr_str
    assert "filename='test.txt'" in repr_str


def test_file_repr_with_file_id():
    file_obj = dspy.File.from_file_id("file-abc", filename="doc.pdf")
    repr_str = repr(file_obj)
    assert "file_id='file-abc'" in repr_str
    assert "filename='doc.pdf'" in repr_str


def test_file_str():
    file_obj = dspy.File.from_bytes(b"test")
    str_repr = str(file_obj)
    assert "<<CUSTOM-TYPE-START-IDENTIFIER>>" in str_repr
    assert "<<CUSTOM-TYPE-END-IDENTIFIER>>" in str_repr


def test_encode_file_to_dict_from_path(sample_text_file):
    result = encode_file_to_dict(sample_text_file)
    assert "file_data" in result
    assert result["file_data"].startswith("data:text/plain;base64,")
    assert "filename" in result


def test_encode_file_to_dict_from_bytes():
    result = encode_file_to_dict(b"test content")
    assert "file_data" in result
    assert result["file_data"].startswith("data:application/octet-stream;base64,")


def test_invalid_file_string():
    with pytest.raises(ValueError, match="Unrecognized"):
        encode_file_to_dict("https://this_is_not_a_file_path")


def test_invalid_dict():
    with pytest.raises(ValueError, match="must contain at least one"):
        dspy.File(invalid="dict")


def test_file_in_signature(sample_text_file):
    signature = "document: dspy.File -> summary: str"
    expected = {"summary": "This is a summary"}
    predictor, lm = setup_predictor(signature, expected)

    file_obj = dspy.File.from_path(sample_text_file)
    result = predictor(document=file_obj)

    assert result.summary == "This is a summary"
    assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 1


def test_file_list_in_signature(sample_text_file):
    class FileListSignature(dspy.Signature):
        documents: list[dspy.File] = dspy.InputField()
        summary: str = dspy.OutputField()

    expected = {"summary": "Multiple files"}
    predictor, lm = setup_predictor(FileListSignature, expected)

    files = [
        dspy.File.from_path(sample_text_file),
        dspy.File.from_file_id("file-123"),
    ]
    result = predictor(documents=files)

    assert result.summary == "Multiple files"
    assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 2


def test_optional_file_field():
    class OptionalFileSignature(dspy.Signature):
        document: dspy.File | None = dspy.InputField()
        output: str = dspy.OutputField()

    predictor, lm = setup_predictor(OptionalFileSignature, {"output": "Hello"})
    result = predictor(document=None)
    assert result.output == "Hello"
    assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 0


def test_save_load_file_signature(sample_text_file):
    signature = "document: dspy.File -> summary: str"
    file_obj = dspy.File.from_path(sample_text_file)
    examples = [dspy.Example(document=file_obj, summary="Test summary")]

    predictor, lm = setup_predictor(signature, {"summary": "A summary"})
    optimizer = dspy.teleprompt.LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)

    with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = dspy.Predict(signature)
        loaded_predictor.load(temp_file.name)

    loaded_predictor(document=dspy.File.from_file_id("file-test"))
    assert count_messages_with_file_pattern(lm.history[-1]["messages"]) == 2


def test_file_frozen():
    file_obj = dspy.File.from_bytes(b"test")
    with pytest.raises((TypeError, ValueError, pydantic.ValidationError)):
        file_obj.file_data = "new_data"


def test_file_with_all_fields():
    file_data_uri = "data:text/plain;base64,dGVzdA=="
    file_obj = dspy.File(file_data=file_data_uri, file_id="file-123", filename="test.txt")
    assert file_obj.file_data == file_data_uri
    assert file_obj.file_id == "file-123"
    assert file_obj.filename == "test.txt"

    formatted = file_obj.format()
    assert formatted[0]["file"]["file_data"] == file_data_uri
    assert formatted[0]["file"]["file_id"] == "file-123"
    assert formatted[0]["file"]["filename"] == "test.txt"


def test_file_path_not_found():
    with pytest.raises(ValueError, match="File not found"):
        dspy.File.from_path("/nonexistent/path/file.txt")


def test_file_custom_mime_type(sample_text_file):
    file_obj = dspy.File.from_path(sample_text_file, mime_type="text/custom")
    assert file_obj.file_data.startswith("data:text/custom;base64,")


def test_file_from_bytes_custom_mime():
    file_obj = dspy.File.from_bytes(b"audio data", mime_type="audio/mp3")
    assert file_obj.file_data.startswith("data:audio/mp3;base64,")


def test_file_data_uri_in_format():
    file_obj = dspy.File.from_bytes(b"test", filename="test.txt", mime_type="text/plain")
    formatted = file_obj.format()
    assert "data:text/plain;base64," in formatted[0]["file"]["file_data"]
