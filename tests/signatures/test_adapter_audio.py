from typing import Dict, List, Optional, Tuple

import pytest
import requests
from io import BytesIO

import dspy
from dspy import Predict
from dspy.utils.dummies import DummyLM
from dspy.adapters.audio_utils import encode_audio
import tempfile
import pydantic

@pytest.fixture
def sample_audio_url():
    return "https://www.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav"

@pytest.fixture
def sample_audio_bytes():
    """Fixture to provide sample audio bytes for testing"""
    url = 'https://www.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav'
    response = requests.get(url)
    response.raise_for_status()
    return response.content

@pytest.fixture
def sample_dspy_audio_download():
    return dspy.Audio.from_url("https://www.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav", download=True)

@pytest.fixture
def sample_dspy_audio_no_download():
    return dspy.Audio.from_url("https://www.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav", download=False)

def count_messages_with_audio_url_pattern(messages):
    pattern = {
        'type': 'audio_url',
        'audio_url': {
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
        "name": "audio_classification",
        "signature": "audio: dspy.Audio, class_labels: List[str] -> probabilities: Dict[str, float]",
        "inputs": {"audio": "https://example.com/sound.wav", "class_labels": ["music", "speech", "noise"]},
        "key_output": "probabilities",
        "expected": {"probabilities": {"music": 0.7, "speech": 0.2, "noise": 0.1}}
    },
    {
        "name": "audio_to_text",
        "signature": "audio: dspy.Audio -> transcription: str",
        "inputs": {"audio": "https://example.com/speech.wav"},
        "key_output": "transcription",
        "expected": {"transcription": "Hello world"}
    },
    {
        "name": "audio_duration",
        "signature": "audio: dspy.Audio -> duration: float",
        "inputs": {"audio": "https://example.com/audio.wav"},
        "key_output": "duration",
        "expected": {"duration": 10.5}
    },
    {
        "name": "multilingual_transcription",
        "signature": "audio: dspy.Audio, languages: List[str] -> transcriptions: Dict[str, str]",
        "inputs": {"audio": "https://example.com/speech.wav", "languages": ["en", "es", "fr"]},
        "key_output": "transcriptions",
        "expected": {"transcriptions": {"en": "Hello", "es": "Hola", "fr": "Bonjour"}}
    }
])
def test_basic_audio_operations(test_case):
    """Consolidated test for basic audio operations"""
    predictor, lm = setup_predictor(test_case["signature"], test_case["expected"])
    
    # Convert string URLs to dspy.Audio objects
    inputs = {k: dspy.Audio.from_url(v) if isinstance(v, str) and k == "audio" else v 
             for k, v in test_case["inputs"].items()}
    
    result = predictor(**inputs)
    
    # Check result based on output field name
    output_field = next(f for f in ["probabilities", "transcription", "duration", "transcriptions"] 
                       if hasattr(result, f))
    assert getattr(result, output_field) == test_case["expected"][test_case["key_output"]]
    assert count_messages_with_audio_url_pattern(lm.history[-1]["messages"]) == 1

@pytest.mark.parametrize("audio_input,description", [
    ("audio_bytes", "audio bytes"),
    ("encoded_audio", "encoded audio string"), 
    ("dspy_audio_download", "dspy.Audio with download=True"),
    ("dspy_audio_no_download", "dspy.Audio without download")
])
def test_audio_input_formats(request, sample_audio_bytes, sample_dspy_audio_download, 
                           sample_dspy_audio_no_download, audio_input, description):
    """Test different input formats for audio fields"""
    signature = "audio: dspy.Audio, class_labels: List[str] -> probabilities: Dict[str, float]"
    expected = {"probabilities": {"music": 0.7, "speech": 0.2, "noise": 0.1}}
    predictor, lm = setup_predictor(signature, expected)

    input_map = {
        "audio_bytes": sample_audio_bytes,
        "encoded_audio": encode_audio(sample_audio_bytes, format="wav"),
        "dspy_audio_download": sample_dspy_audio_download,
        "dspy_audio_no_download": sample_dspy_audio_no_download
    }
    
    actual_input = input_map[audio_input]
    # TODO: Support the cases without direct dspy.Audio coercion
    if audio_input in ["audio_bytes", "encoded_audio"]:
        pytest.xfail(f"{description} not fully supported without dspy.from_bytes")

    result = predictor(audio=actual_input, class_labels=["music", "speech", "noise"])
    assert result.probabilities == expected["probabilities"]
    assert count_messages_with_audio_url_pattern(lm.history[-1]["messages"]) == 1

def test_predictor_save_load(sample_audio_url, sample_audio_bytes):
    """Test saving and loading predictors with audio fields"""
    signature = "audio: dspy.Audio -> transcription: str"
    examples = [
        dspy.Example(audio=dspy.Audio.from_url(sample_audio_url), transcription="Example 1"),
        dspy.Example(audio=dspy.Audio.from_bytes(sample_audio_bytes), transcription="Example 2"),
    ]
    
    predictor, lm = setup_predictor(signature, {"transcription": "Hello world"})
    optimizer = dspy.teleprompt.LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)

    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = dspy.Predict(signature)
        loaded_predictor.load(temp_file.name)
    
    result = loaded_predictor(audio=dspy.Audio.from_url("https://example.com/audio.wav"))
    assert count_messages_with_audio_url_pattern(lm.history[-1]["messages"]) == 2
    assert "<DSPY_AUDIO_START>" not in str(lm.history[-1]["messages"])

def test_save_load_complex_default_types():
    """Test saving and loading predictors with complex default types (lists of audio)"""
    examples = [
        dspy.Example(
            audio_list=[
                dspy.Audio.from_url("https://example.com/audio1.wav"),
                dspy.Audio.from_url("https://example.com/audio2.wav")
            ],
            transcription="Example 1"
        ).with_inputs("audio_list"),
    ]

    class ComplexTypeSignature(dspy.Signature):
        audio_list: List[dspy.Audio] = dspy.InputField(desc="A list of audio files")
        transcription: str = dspy.OutputField(desc="A transcription for the audio list")

    predictor, lm = setup_predictor(ComplexTypeSignature, {"transcription": "Multiple audio files"})
    optimizer = dspy.teleprompt.LabeledFewShot(k=1)
    compiled_predictor = optimizer.compile(student=predictor, trainset=examples, sample=False)

    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix=".json") as temp_file:
        compiled_predictor.save(temp_file.name)
        loaded_predictor = dspy.Predict(ComplexTypeSignature)
        loaded_predictor.load(temp_file.name)
    
    result = loaded_predictor(**examples[0].inputs())
    assert result.transcription == "Multiple audio files"
    assert str(lm.history[-1]["messages"]).count("'url'") == 4
    assert "<DSPY_AUDIO_START>" not in str(lm.history[-1]["messages"])

class BasicAudioSignature(dspy.Signature):
    """Basic signature with a single audio input"""
    audio: dspy.Audio = dspy.InputField()
    output: str = dspy.OutputField()

class AudioListSignature(dspy.Signature):
    """Signature with a list of audio inputs"""
    audio_list: List[dspy.Audio] = dspy.InputField()
    output: str = dspy.OutputField()

@pytest.mark.parametrize("test_case", [
    {
        "name": "basic_dspy_signature",
        "signature_class": BasicAudioSignature,
        "inputs": {
            "audio": "https://example.com/audio.wav"
        },
        "expected": {"output": "An audio file"},
        "expected_audio_urls": 2
    },
    {
        "name": "list_dspy_signature",
        "signature_class": AudioListSignature,
        "inputs": {
            "audio_list": ["https://example.com/audio1.wav", "https://example.com/audio2.wav"]
        },
        "expected": {"output": "Multiple audio files"},
        "expected_audio_urls": 4
    }
])
def test_save_load_complex_types(test_case):
    """Test saving and loading predictors with complex types"""
    signature_cls = test_case["signature_class"]
    
    # Convert string URLs to dspy.Audio objects in input
    processed_input = {}
    for key, value in test_case["inputs"].items():
        if isinstance(value, str) and "http" in value:
            processed_input[key] = dspy.Audio.from_url(value)
        elif isinstance(value, list) and value and isinstance(value[0], str):
            processed_input[key] = [dspy.Audio.from_url(url) for url in value]
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
    
    # Verify correct number of audio URLs in messages
    assert count_messages_with_audio_url_pattern(lm.history[-1]["messages"]) == test_case["expected_audio_urls"]
    assert "<DSPY_AUDIO_START>" not in str(lm.history[-1]["messages"])

def test_save_load_pydantic_model():
    """Test saving and loading predictors with pydantic models"""
    class AudioModel(pydantic.BaseModel):
        audio: dspy.Audio
        audio_list: Optional[List[dspy.Audio]] = None
        output: str

    class PydanticSignature(dspy.Signature):
        model_input: AudioModel = dspy.InputField()
        output: str = dspy.OutputField()

    # Create model instance
    model_input = AudioModel(
        audio=dspy.Audio.from_url("https://example.com/audio1.wav"),
        audio_list=[dspy.Audio.from_url("https://example.com/audio2.wav")],
        output="Multiple audio files"
    )

    # Create example and predictor
    examples = [
        dspy.Example(model_input=model_input, output="Multiple audio files").with_inputs("model_input")
    ]

    predictor, lm = setup_predictor(PydanticSignature, {"output": "Multiple audio files"})
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
    assert result.output == "Multiple audio files"
    assert count_messages_with_audio_url_pattern(lm.history[-1]["messages"]) == 4
    assert "<DSPY_AUDIO_START>" not in str(lm.history[-1]["messages"])

def test_optional_audio_field():
    """Test that optional audio fields are not required"""
    class OptionalAudioSignature(dspy.Signature):
        audio: Optional[dspy.Audio] = dspy.InputField()
        output: str = dspy.OutputField()

    predictor, lm = setup_predictor(OptionalAudioSignature, {"output": "Hello"})
    result = predictor(audio=None)
    assert result.output == "Hello"
    assert count_messages_with_audio_url_pattern(lm.history[-1]["messages"]) == 0

def test_audio_repr():
    """Test string representation of Audio objects"""
    url_audio = dspy.Audio.from_url("https://example.com/audio.wav", download=False)
    assert str(url_audio) == "<DSPY_AUDIO_START>https://example.com/audio.wav<DSPY_AUDIO_END>"
    assert repr(url_audio) == "Audio(url='https://example.com/audio.wav')"
    
    sample_bytes = b"test audio data"
    bytes_audio = dspy.Audio.from_bytes(sample_bytes, format="wav")
    assert str(bytes_audio).startswith("<DSPY_AUDIO_START>data:audio/wav;base64,")
    assert str(bytes_audio).endswith("<DSPY_AUDIO_END>")
    assert "base64" in str(bytes_audio) 