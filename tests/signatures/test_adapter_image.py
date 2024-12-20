from typing import Dict, List, Optional, Tuple

import pytest
from PIL import Image as PILImage
import requests
from io import BytesIO

import dspy
from dspy import Predict
from dspy.utils.dummies import DummyLM
import tempfile
import pydantic

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



def messages_contain_image_url_pattern(messages, n=1):
    pattern = {
        'type': 'image_url',
        'image_url': {
            'url': lambda x: isinstance(x, str)
        }
    }
    
    try:
        # Helper function to check nested dict matches pattern
        def check_pattern(obj, pattern):
            if isinstance(pattern, dict):
                if not isinstance(obj, dict):
                    return False
                return all(
                    k in obj and check_pattern(obj[k], v) 
                    for k, v in pattern.items()
                )
            if callable(pattern):
                return pattern(obj)
            return obj == pattern
            
        # Look for pattern in any nested dict and count occurrences
        def count_patterns(obj, pattern):
            count = 0
            if check_pattern(obj, pattern):
                count += 1
            if isinstance(obj, dict):
                count += sum(count_patterns(v, pattern) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                count += sum(count_patterns(v, pattern) for v in obj)
            return count
            
        return count_patterns(messages, pattern) == n
    except Exception:
        return False
    
def test_probabilistic_classification():
    class ProbabilisticClassificationSignature(dspy.Signature):
        image: dspy.Image = dspy.InputField(desc="An image to classify")
        class_labels: List[str] = dspy.InputField(desc="Possible class labels")
        probabilities: Dict[str, float] = dspy.OutputField(desc="Probability distribution over the class labels")

    expected = {"dog": 0.8, "cat": 0.1, "bird": 0.1}
    lm = DummyLM([{"probabilities": str(expected)}])
    dspy.settings.configure(lm=lm)

    predictor = dspy.Predict(ProbabilisticClassificationSignature)
    result = predictor(
        image=dspy.Image.from_url("https://example.com/dog.jpg"),
        class_labels=["dog", "cat", "bird"]
    )

    assert result.probabilities == expected
    assert messages_contain_image_url_pattern(lm.history[-1]["messages"])


def test_inline_classification():
    signature = "image: dspy.Image, class_labels: List[str] -> probabilities: Dict[str, float]"
    
    expected = {"dog": 0.8, "cat": 0.1, "bird": 0.1}
    lm = DummyLM([{"probabilities": str(expected)}])
    dspy.settings.configure(lm=lm)

    predictor = dspy.Predict(signature)
    result = predictor(
        image=dspy.Image.from_url("https://example.com/dog.jpg"),
        class_labels=["dog", "cat", "bird"]
    )

    assert result.probabilities == expected
    assert messages_contain_image_url_pattern(lm.history[-1]["messages"])


def test_image_to_code():
    class ImageToCodeSignature(dspy.Signature):
        ui_image: dspy.Image = dspy.InputField(desc="An image of a user interface")
        target_language: str = dspy.InputField(desc="Programming language for the generated code")
        generated_code: str = dspy.OutputField(desc="Code that replicates the UI shown in the image")

    expected_code = "<button>Click me</button>"
    lm = DummyLM([{"generated_code": expected_code}])
    dspy.settings.configure(lm=lm)

    predictor = dspy.Predict(ImageToCodeSignature)
    result = predictor(
        ui_image=dspy.Image.from_url("https://example.com/button.png"),
        target_language="HTML"
    )

    assert result.generated_code == expected_code
    assert messages_contain_image_url_pattern(lm.history[-1]["messages"])

def test_bbox_image():
    class BBOXImageSignature(dspy.Signature):
        image: dspy.Image = dspy.InputField(desc="The original image to annotate")
        bboxes: List[Tuple[int, int, int, int]] = dspy.OutputField(
            desc="List of bounding boxes with coordinates (x1, y1, x2, y2)"
        )

    expected_bboxes = [(10, 20, 30, 40), (50, 60, 70, 80)]
    lm = DummyLM([{"bboxes": str(expected_bboxes)}])
    dspy.settings.configure(lm=lm)

    predictor = dspy.Predict(BBOXImageSignature)
    result = predictor(image=dspy.Image.from_url("https://example.com/image.jpg"))

    assert result.bboxes == expected_bboxes
    assert messages_contain_image_url_pattern(lm.history[-1]["messages"])


def test_multilingual_caption():
    class MultilingualCaptionSignature(dspy.Signature):
        image: dspy.Image = dspy.InputField(desc="An image to generate captions for")
        languages: List[str] = dspy.InputField(
            desc="List of language codes for the captions (e.g., ['en', 'es', 'fr'])"
        )
        captions: Dict[str, str] = dspy.OutputField(
            desc="Captions in different languages keyed by language code"
        )

    expected_captions = {
        "en": "A golden retriever",
        "es": "Un golden retriever",
        "fr": "Un golden retriever"
    }
    lm = DummyLM([{"captions": str(expected_captions)}])
    dspy.settings.configure(lm=lm)

    predictor = dspy.Predict(MultilingualCaptionSignature)
    result = predictor(
        image=dspy.Image.from_url("https://example.com/dog.jpg"),
        languages=["en", "es", "fr"]
    )

    assert result.captions == expected_captions
    assert messages_contain_image_url_pattern(lm.history[-1]["messages"])

from dspy.adapters.image_utils import encode_image

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
    expected = {"dog": 0.8, "cat": 0.1, "bird": 0.1}
    lm = DummyLM([{"probabilities": str(expected)}])
    dspy.settings.configure(lm=lm)
    predictor = dspy.Predict(signature)

    # Map input type to actual input
    input_map = {
        "pil_image": sample_pil_image,
        "encoded_pil_image": encode_image(sample_pil_image),
        "dspy_image_download": sample_dspy_image_download,
        "dspy_image_no_download": sample_dspy_image_no_download
    }
    
    actual_input = input_map[image_input]
    
    # TODO(isaacbmiller): Support the cases without direct dspy.Image coercion
    if image_input == "pil_image":
        pytest.xfail("PIL images not fully supported without dspy.from_PIL")
    if image_input == "encoded_pil_image":
        pytest.xfail("encoded PIL images not fully supported without dspy.from_PIL")

    result = predictor(
        image=actual_input,
        class_labels=["dog", "cat", "bird"]
    )
    
    assert result.probabilities == expected
    assert messages_contain_image_url_pattern(lm.history[-1]["messages"])

def test_invalid_image_input(sample_url):
    """Test that using a string input with str annotation fails"""
    signature = "image: str, class_labels: List[str] -> probabilities: Dict[str, float]"
    lm = DummyLM([{"probabilities": "{}"}])
    dspy.settings.configure(lm=lm)
    predictor = dspy.Predict(signature)

    result = predictor(
        image=sample_url,
        class_labels=["dog", "cat", "bird"]
    )
    assert not messages_contain_image_url_pattern(lm.history[-1]["messages"])

def test_predictor_save_load(sample_url, sample_pil_image):
    signature = "image: dspy.Image -> caption: str"
    examples = [
        dspy.Example(image=dspy.Image.from_url(sample_url), caption="Example 1"),
        dspy.Example(image=sample_pil_image, caption="Example 2"),
    ]
    active_example = dspy.Example(image=dspy.Image.from_url("https://example.com/dog.jpg"))

    lm = DummyLM([{"caption": "A golden retriever"}])
    dspy.settings.configure(lm=lm)

    predictor = dspy.Predict(signature)
    optimizer = dspy.teleprompt.LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)
    print(compiled_predictor.demos)
    # Test dump state with save verbose = True and False
    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = dspy.Predict(signature)
        loaded_predictor.load(temp_file.name)
    print(loaded_predictor.demos)
    
    result = loaded_predictor(image=active_example["image"])
    print(result)
    assert messages_contain_image_url_pattern(lm.history[-1]["messages"], n=2)
    print(lm.history[-1]["messages"])
    assert "<DSPY_IMAGE_START>" not in str(lm.history[-1]["messages"])

def test_save_load_complex_default_types():
    examples = [
        dspy.Example(image_list=[dspy.Image.from_url("https://example.com/dog.jpg"), dspy.Image.from_url("https://example.com/cat.jpg")], caption="Example 1").with_inputs("image_list"),
    ]

    class ComplexTypeSignature(dspy.Signature):
        image_list: List[dspy.Image] = dspy.InputField(desc="A list of images")
        caption: str = dspy.OutputField(desc="A caption for the image list")

    lm = DummyLM([{"caption": "A list of images"}, {"caption": "A list of images"}])
    dspy.settings.configure(lm=lm)

    predictor = dspy.Predict(ComplexTypeSignature)
    result = predictor(**examples[0].inputs())
    
    print(lm.history[-1]["messages"])
    assert "<DSPY_IMAGE_START>" not in str(lm.history[-1]["messages"])
    assert str(lm.history[-1]["messages"]).count("'url'") == 2

    optimizer = dspy.teleprompt.LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)
    print(compiled_predictor.demos)

    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file:
        print("compiled_predictor state: ", compiled_predictor.dump_state())
        compiled_predictor.save(temp_file.name)
        loaded_predictor = dspy.Predict(ComplexTypeSignature)
        loaded_predictor.load(temp_file.name)
    
    print("loaded_predictor state: ", loaded_predictor.dump_state())
    result = loaded_predictor(**examples[0].inputs())
    assert result.caption == "A list of images"
    assert str(lm.history[-1]["messages"]).count("'url'") == 4

def test_save_load_complex_pydantic_types():
    """Test saving and loading predictors with pydantic models containing image fields"""
    class ImageModel(pydantic.BaseModel):
        image: dspy.Image
        label: str

    class ComplexPydanticSignature(dspy.Signature):
        model_input: ImageModel = dspy.InputField(desc="A pydantic model containing an image")
        caption: str = dspy.OutputField(desc="A caption for the image")

    example_model = ImageModel(
        image=dspy.Image.from_url("https://example.com/dog.jpg"),
        label="dog"
    )
    examples = [
        dspy.Example(model_input=example_model, caption="A dog").with_inputs("model_input")
    ]

    lm = DummyLM([{"caption": "A dog photo"}, {"caption": "A dog photo"}])
    dspy.settings.configure(lm=lm)

    predictor = dspy.Predict(ComplexPydanticSignature)
    result = predictor(model_input=example_model)
    assert result.caption == "A dog photo"
    assert messages_contain_image_url_pattern(lm.history[-1]["messages"])

    optimizer = dspy.teleprompt.LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)

    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = dspy.Predict(ComplexPydanticSignature)
        loaded_predictor.load(temp_file.name)

    result = loaded_predictor(model_input=example_model)
    lm.inspect_history()
    assert result.caption == "A dog photo"
    assert messages_contain_image_url_pattern(lm.history[-1]["messages"], n=2)

def test_image_repr():
    """Test string representation of Image objects with both URLs and PIL images"""
    # Test URL-based image repr and str
    url_image = dspy.Image.from_url("https://example.com/dog.jpg", download=False)
    assert str(url_image) == "<DSPY_IMAGE_START>https://example.com/dog.jpg<DSPY_IMAGE_END>"
    assert repr(url_image) == "Image(url='https://example.com/dog.jpg')"
    
    # Test PIL image repr and str
    sample_pil = PILImage.new('RGB', (60, 30), color='red')
    pil_image = dspy.Image.from_PIL(sample_pil)
    # Test str() behavior
    assert str(pil_image).startswith("<DSPY_IMAGE_START>data:image/png;base64,")
    assert str(pil_image).endswith("<DSPY_IMAGE_END>")
    # Test repr() behavior
    repr_str = repr(pil_image)
    assert repr_str.startswith("Image(url=data:image/...base64,<IMAGE_BASE_64_ENCODED(")
    assert repr_str.endswith(")>)")
    assert "base64" in str(pil_image)

def test_image_optional_input():
    """Test behavior when optional image inputs are missing"""
    class OptionalImageSignature(dspy.Signature):
        image: Optional[dspy.Image] = dspy.InputField(desc="An optional image input")
        text: str = dspy.InputField(desc="A text input")
        output: str = dspy.OutputField(desc="The output text")

    lm = DummyLM([{"output": "Text only: hello"}, {"output": "Image and text: hello with image"}])
    dspy.settings.configure(lm=lm)

    predictor = dspy.Predict(OptionalImageSignature)
    
    # Test with missing image
    result = predictor(image=None, text="hello")
    assert result.output == "Text only: hello"
    assert not messages_contain_image_url_pattern(lm.history[-1]["messages"])

    # Test with image present
    result = predictor(
        image=dspy.Image.from_url("https://example.com/image.jpg"),
        text="hello"
    )
    lm.inspect_history()
    
    assert result.output == "Image and text: hello with image"
    assert messages_contain_image_url_pattern(lm.history[-1]["messages"])

# Tests to write:
# complex image types
# Return "None" when dspy.Image is missing for the input, similarly to str input fields
# JSON adapter
