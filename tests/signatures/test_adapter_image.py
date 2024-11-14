import datetime
from typing import Dict, List, Tuple

import pytest
from PIL import Image
import requests
from io import BytesIO

import dspy
from dspy import Predict
from dspy.utils.dummies import DummyLM
import tempfile

@pytest.fixture
def sample_pil_image():
    """Fixture to provide a sample image for testing"""
    url = 'https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg'
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

@pytest.fixture
def sample_dspy_image_download():
    return dspy.Image.from_url("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg", download=True)

@pytest.fixture
def sample_url():
    return "https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg"

@pytest.fixture
def sample_dspy_image_no_download():
    return dspy.Image.from_url("https://images.dog.ceo/breeds/dane-great/n02109047_8912.jpg", download=False)



def messages_contain_image_url_pattern(messages):
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
            
        # Look for pattern in any nested dict
        def find_pattern(obj, pattern):
            if check_pattern(obj, pattern):
                return True
            if isinstance(obj, dict):
                return any(find_pattern(v, pattern) for v in obj.values())
            if isinstance(obj, (list, tuple)):
                return any(find_pattern(v, pattern) for v in obj)
            return False
            
        return find_pattern(messages, pattern)
    except:
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

def test_image_input_formats(sample_pil_image, sample_dspy_image_download, sample_dspy_image_no_download):
    """Test different input formats for image fields"""
    from dspy.adapters.image_utils import encode_image

    signature = "image: dspy.Image, class_labels: List[str] -> probabilities: Dict[str, float]"
    expected = {"dog": 0.8, "cat": 0.1, "bird": 0.1}
    lm = DummyLM([{"probabilities": str(expected)}] * 4)  # Need multiple responses for different tests
    dspy.settings.configure(lm=lm)
    predictor = dspy.Predict(signature)

    # Test PIL Image input
    result = predictor(image=sample_pil_image, class_labels=["dog", "cat", "bird"])
    assert result.probabilities == expected
    assert messages_contain_image_url_pattern(lm.history[-1]["messages"])
    # Test encoded string input
    encoded_image = encode_image(sample_pil_image)
    result = predictor(image=encoded_image, class_labels=["dog", "cat", "bird"])
    assert result.probabilities == expected
    assert messages_contain_image_url_pattern(lm.history[-1]["messages"])

    # Test dspy.Image with download=True
    result = predictor(
        image=sample_dspy_image_download,
        class_labels=["dog", "cat", "bird"]
    )
    assert result.probabilities == expected
    assert messages_contain_image_url_pattern(lm.history[-1]["messages"])
    # Test dspy.Image without download
    result = predictor(
        image=sample_dspy_image_no_download,
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
    assert messages_contain_image_url_pattern(lm.history[-1]["messages"])
    print(lm.history[-1]["messages"])
    assert False

def test_save_load_complex_types():
    pass
    # class ComplexTypeSignature(dspy.Signature):
    #     image_list: List[dspy.Image] = dspy.InputField(desc="A list of images")
    #     caption: str = dspy.OutputField(desc="A caption for the image list")

    # lm = DummyLM([{"caption": "A list of images"}])
    # dspy.settings.configure(lm=lm)

    # predictor = dspy.Predict(ComplexTypeSignature)
    # result = predictor(image_list=[dspy.Image.from_url("https://example.com/dog.jpg")])
    # assert isinstance(result.caption, str)
