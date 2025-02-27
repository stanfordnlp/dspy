from typing import Dict, List, Optional, Tuple

import pytest
from PIL import Image as PILImage
import requests
from io import BytesIO

import dspy
from dspy import Predict
from dspy.utils.dummies import DummyLM
from dspy.adapters.types.image import encode_image
import tempfile
import pydantic
import os


@pytest.fixture
def sample_pil_image():
    """Fixture to provide a sample image for testing"""
    url = 'https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg'
    response = requests.get(url)
    response.raise_for_status()
    return PILImage.open(BytesIO(response.content))

@pytest.fixture
def sample_dspy_image_download():
    return dspy.Image.from_url("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg", download=True)

@pytest.fixture
def sample_url():
    return "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"

@pytest.fixture
def sample_dspy_image_no_download():
    return dspy.Image.from_url("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg", download=False)

def count_messages_with_image_url_pattern(messages):
    pattern = {
        'type': 'image_url',
        'image_url': {
            'url': lambda x: isinstance(x, str)
        }
    }
    
    try:
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
            if isinstance(obj, (list, tuple)):
                count += sum(count_patterns(v, pattern) for v in obj)
            return count
            
        return count_patterns(messages, pattern)
    except Exception:
        return 0

def setup_predictor(signature, expected_output):
    """Helper to set up a predictor with DummyLM"""
    lm = DummyLM([expected_output])
    dspy.settings.configure(lm=lm)
    return dspy.Predict(signature), lm

@pytest.mark.parametrize("test_case", [
    {
        "name": "probabilistic_classification",
        "signature": "image: dspy.Image, class_labels: List[str] -> probabilities: Dict[str, float]",
        "inputs": {"image": "https://example.com/dog.jpg", "class_labels": ["dog", "cat", "bird"]},
        "key_output": "probabilities",
        "expected": {"probabilities": {"dog": 0.8, "cat": 0.1, "bird": 0.1}}
    },
    {
        "name": "image_to_code",
        "signature": "ui_image: dspy.Image, target_language: str -> generated_code: str",
        "inputs": {"ui_image": "https://example.com/button.png", "target_language": "HTML"},
        "key_output": "generated_code",
        "expected": {"generated_code": "<button>Click me</button>"}
    },
    {
        "name": "bbox_detection",
        "signature": "image: dspy.Image -> bboxes: List[Tuple[int, int, int, int]]",
        "inputs": {"image": "https://example.com/image.jpg"},
        "key_output": "bboxes",
        "expected": {"bboxes": [(10, 20, 30, 40), (50, 60, 70, 80)]}
    },
    {
        "name": "multilingual_caption",
        "signature": "image: dspy.Image, languages: List[str] -> captions: Dict[str, str]",
        "inputs": {"image": "https://example.com/dog.jpg", "languages": ["en", "es", "fr"]},
        "key_output": "captions",
        "expected": {"captions": {"en": "A golden retriever", "es": "Un golden retriever", "fr": "Un golden retriever"}}
    }
])
def test_basic_image_operations(test_case):
    """Consolidated test for basic image operations"""
    predictor, lm = setup_predictor(test_case["signature"], test_case["expected"])
    
    # Convert string URLs to dspy.Image objects
    inputs = {k: dspy.Image.from_url(v) if isinstance(v, str) and k in ["image", "ui_image"] else v 
             for k, v in test_case["inputs"].items()}
    
    result = predictor(**inputs)
    
    # Check result based on output field name
    output_field = next(f for f in ["probabilities", "generated_code", "bboxes", "captions"] 
                       if hasattr(result, f))
    assert getattr(result, output_field) == test_case["expected"][test_case["key_output"]]
    assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == 1

@pytest.mark.parametrize("image_input,description", [
    ("pil_image", "PIL Image"),
    ("encoded_pil_image", "encoded PIL image string"), 
    ("dspy_image_download", "dspy.Image with download=True"),
    ("dspy_image_no_download", "dspy.Image without download")
])
def test_image_input_formats(request, sample_pil_image, sample_dspy_image_download, 
                           sample_dspy_image_no_download, image_input, description):
    """Test different input formats for image fields"""
    signature = "image: dspy.Image, class_labels: List[str] -> probabilities: Dict[str, float]"
    expected = {"probabilities": {"dog": 0.8, "cat": 0.1, "bird": 0.1}}
    predictor, lm = setup_predictor(signature, expected)

    input_map = {
        "pil_image": sample_pil_image,
        "encoded_pil_image": encode_image(sample_pil_image),
        "dspy_image_download": sample_dspy_image_download,
        "dspy_image_no_download": sample_dspy_image_no_download
    }
    
    actual_input = input_map[image_input]
    # TODO(isaacbmiller): Support the cases without direct dspy.Image coercion
    if image_input in ["pil_image", "encoded_pil_image"]:
        pytest.xfail(f"{description} not fully supported without dspy.from_PIL")

    result = predictor(image=actual_input, class_labels=["dog", "cat", "bird"])
    assert result.probabilities == expected["probabilities"]
    assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == 1

def test_predictor_save_load(sample_url, sample_pil_image):
    """Test saving and loading predictors with image fields"""
    signature = "image: dspy.Image -> caption: str"
    examples = [
        dspy.Example(image=dspy.Image.from_url(sample_url), caption="Example 1"),
        dspy.Example(image=sample_pil_image, caption="Example 2"),
    ]
    
    predictor, lm = setup_predictor(signature, {"caption": "A golden retriever"})
    optimizer = dspy.teleprompt.LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)

    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = dspy.Predict(signature)
        loaded_predictor.load(temp_file.name)
    
    result = loaded_predictor(image=dspy.Image.from_url("https://example.com/dog.jpg"))
    assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == 2
    assert "<DSPY_IMAGE_START>" not in str(lm.history[-1]["messages"])

def test_save_load_complex_default_types():
    """Test saving and loading predictors with complex default types (lists of images)"""
    examples = [
        dspy.Example(
            image_list=[
                dspy.Image.from_url("https://example.com/dog.jpg"),
                dspy.Image.from_url("https://example.com/cat.jpg")
            ],
            caption="Example 1"
        ).with_inputs("image_list"),
    ]

    class ComplexTypeSignature(dspy.Signature):
        image_list: List[dspy.Image] = dspy.InputField(desc="A list of images")
        caption: str = dspy.OutputField(desc="A caption for the image list")

    predictor, lm = setup_predictor(ComplexTypeSignature, {"caption": "A list of images"})
    optimizer = dspy.teleprompt.LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)

    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = dspy.Predict(ComplexTypeSignature)
        loaded_predictor.load(temp_file.name)
    
    result = loaded_predictor(**examples[0].inputs())
    assert result.caption == "A list of images"
    assert str(lm.history[-1]["messages"]).count("'url'") == 4
    assert "<DSPY_IMAGE_START>" not in str(lm.history[-1]["messages"])

class BasicImageSignature(dspy.Signature):
    """Basic signature with a single image input"""
    image: dspy.Image = dspy.InputField()
    output: str = dspy.OutputField()

class ImageListSignature(dspy.Signature):
    """Signature with a list of images input"""
    image_list: List[dspy.Image] = dspy.InputField()
    output: str = dspy.OutputField()

@pytest.mark.parametrize("test_case", [
    {
        "name": "basic_dspy_signature",
        "signature_class": BasicImageSignature,
        "inputs": {
            "image": "https://example.com/dog.jpg"
        },
        "expected": {"output": "A dog photo"},
        "expected_image_urls": 2
    },
    {
        "name": "list_dspy_signature",
        "signature_class": ImageListSignature,
        "inputs": {
            "image_list": ["https://example.com/dog.jpg", "https://example.com/cat.jpg"]
        },
        "expected": {"output": "Multiple photos"},
        "expected_image_urls": 4
    }
])
def test_save_load_complex_types(test_case):
    """Test saving and loading predictors with complex types"""
    signature_cls = test_case["signature_class"]
    
    # Convert string URLs to dspy.Image objects in input
    processed_input = {}
    for key, value in test_case["inputs"].items():
        if isinstance(value, str) and "http" in value:
            processed_input[key] = dspy.Image.from_url(value)
        elif isinstance(value, list) and value and isinstance(value[0], str):
            processed_input[key] = [dspy.Image.from_url(url) for url in value]
        else:
            processed_input[key] = value
    
    # Create example and predictor
    examples = [
        dspy.Example(**processed_input, **test_case["expected"]).with_inputs(*processed_input.keys())
    ]
    
    predictor, lm = setup_predictor(signature_cls, test_case["expected"])
    optimizer = dspy.teleprompt.LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)
    
    # Test save and load
    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = dspy.Predict(signature_cls)
        loaded_predictor.load(temp_file.name)
    
    # Run prediction
    result = loaded_predictor(**processed_input)
    
    # Verify output matches expected
    for key, value in test_case["expected"].items():
        assert getattr(result, key) == value
    
    # Verify correct number of image URLs in messages
    assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == test_case["expected_image_urls"]
    assert "<DSPY_IMAGE_START>" not in str(lm.history[-1]["messages"])

def test_save_load_pydantic_model():
    """Test saving and loading predictors with pydantic models"""
    class ImageModel(pydantic.BaseModel):
        image: dspy.Image
        image_list: Optional[List[dspy.Image]] = None
        output: str

    class PydanticSignature(dspy.Signature):
        model_input: ImageModel = dspy.InputField()
        output: str = dspy.OutputField()

    # Create model instance
    model_input = ImageModel(
        image=dspy.Image.from_url("https://example.com/dog.jpg"),
        image_list=[dspy.Image.from_url("https://example.com/cat.jpg")],
        output="Multiple photos"
    )

    # Create example and predictor
    examples = [
        dspy.Example(model_input=model_input, output="Multiple photos").with_inputs("model_input")
    ]

    predictor, lm = setup_predictor(PydanticSignature, {"output": "Multiple photos"})
    optimizer = dspy.teleprompt.LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)

    # Test save and load
    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = dspy.Predict(PydanticSignature)
        loaded_predictor.load(temp_file.name)

    # Run prediction
    result = loaded_predictor(model_input=model_input)

    # Verify output matches expected
    assert result.output == "Multiple photos"
    assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == 4
    assert "<DSPY_IMAGE_START>" not in str(lm.history[-1]["messages"])

def test_optional_image_field():
    """Test that optional image fields are not required"""
    class OptionalImageSignature(dspy.Signature):
        image: Optional[dspy.Image] = dspy.InputField()
        output: str = dspy.OutputField()

    predictor, lm = setup_predictor(OptionalImageSignature, {"output": "Hello"})
    result = predictor(image=None)
    assert result.output == "Hello"
    assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == 0


def test_pdf_url_support():
    """Test support for PDF files from URLs"""
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

    # Create a dspy.Image object from the PDF URL with download=True
    pdf_image = dspy.Image.from_url(pdf_url, download=True)

    # The data URI should contain application/pdf in the MIME type
    assert "data:application/pdf" in pdf_image.url
    assert ";base64," in pdf_image.url

    # Test using it in a predictor
    class PDFSignature(dspy.Signature):
        document: dspy.Image = dspy.InputField(desc="A PDF document")
        summary: str = dspy.OutputField(desc="A summary of the PDF")

    predictor, lm = setup_predictor(PDFSignature, {"summary": "This is a dummy PDF"})
    result = predictor(document=pdf_image)

    assert result.summary == "This is a dummy PDF"
    assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == 1

    # Ensure the URL was properly expanded in messages
    messages_str = str(lm.history[-1]["messages"])
    assert "application/pdf" in messages_str


def test_different_mime_types():
    """Test support for different file types and MIME type detection"""
    # Test with various file types
    file_urls = {
        "pdf": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        "image": "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg",
    }

    expected_mime_types = {
        "pdf": "application/pdf",
        "image": "image/jpeg",
    }

    for file_type, url in file_urls.items():
        # Download and encode
        encoded = encode_image(url, download_images=True)

        # Check for correct MIME type in the encoded data - using 'in' instead of startswith
        # to account for possible parameters in the MIME type
        assert f"data:{expected_mime_types[file_type]}" in encoded
        assert ";base64," in encoded


def test_mime_type_from_response_headers():
    """Test that MIME types from response headers are correctly used"""
    # This URL returns proper Content-Type header
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"

    # Make an actual request to get the content type from headers
    response = requests.get(pdf_url)
    expected_mime_type = response.headers.get("Content-Type", "")

    # Should be application/pdf or similar
    assert "pdf" in expected_mime_type.lower()

    # Encode with download to test MIME type from headers
    encoded = encode_image(pdf_url, download_images=True)

    # The encoded data should contain the correct MIME type
    assert "application/pdf" in encoded
    assert ";base64," in encoded


def test_pdf_from_file():
    """Test handling a PDF file from disk"""
    # Download a PDF to a temporary file
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    response = requests.get(pdf_url)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    try:
        # Create a dspy.Image from the file
        pdf_image = dspy.Image.from_file(tmp_file_path)

        # Check that the MIME type is correct
        assert "data:application/pdf" in pdf_image.url
        assert ";base64," in pdf_image.url

        # Test the image in a predictor
        class FilePDFSignature(dspy.Signature):
            document: dspy.Image = dspy.InputField(desc="A PDF document from file")
            summary: str = dspy.OutputField(desc="A summary of the PDF")

        predictor, lm = setup_predictor(FilePDFSignature, {"summary": "This is a PDF from file"})
        result = predictor(document=pdf_image)

        assert result.summary == "This is a PDF from file"
        assert count_messages_with_image_url_pattern(lm.history[-1]["messages"]) == 1
    finally:
        # Clean up the temporary file
        try:
            os.unlink(tmp_file_path)
        except:
            pass


def test_image_repr():
    """Test string representation of Image objects"""
    url_image = dspy.Image.from_url("https://example.com/dog.jpg", download=False)
    assert str(url_image) == "<DSPY_IMAGE_START>https://example.com/dog.jpg<DSPY_IMAGE_END>"
    assert repr(url_image) == "Image(url='https://example.com/dog.jpg')"
    
    sample_pil = PILImage.new('RGB', (60, 30), color='red')
    pil_image = dspy.Image.from_PIL(sample_pil)
    assert str(pil_image).startswith("<DSPY_IMAGE_START>data:image/png;base64,")
    assert str(pil_image).endswith("<DSPY_IMAGE_END>")
    assert "base64" in str(pil_image)