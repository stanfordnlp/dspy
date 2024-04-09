# Implementation of the Image class


import numpy as np
from pathlib import Path
import PIL.Image as PILImage
from dsp.primitives.vision import Image
import pytest
import io
from unittest.mock import MagicMock, patch

@pytest.fixture
def image():
    return Image(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8), encoding='png')
  
def test_image_initialization_with_numpy_array():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    img = Image(arr)
    assert isinstance(img.array, np.ndarray)
    assert img.array.shape == (10, 10, 3)

def test_image_initialization_with_base64(image):
    
    base64_str = image.base64
    img = Image(base64_str)
    assert img.base64 == base64_str

def test_image_initialization_with_file_path(tmp_path, image):
    file_path = Path(tmp_path) / 'image.png'
    image.pil.save(str(file_path))
    img_from_file = Image(str(file_path))
    assert img_from_file.path == str(file_path)
    assert img_from_file.pil.size == image.size
    

def test_image_initialization_with_url(image):
    with patch('dsp.primitives.vision.urlopen') as mock_urlopen:
        class MockResponse:
            def __init__(self, data):
                self.data = data
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_value, traceback):
                pass
            def read(self):
                return self.data
        mock_urlopen.return_value = MockResponse(image.pil.tobytes())

        # Create an Image object with the URL
        url = "http://someurl.com/image.png"
        img = Image(url=url)

        # Verify the resulting Image object
        assert img.pil.tobytes() == image.pil.tobytes()
        assert img.size == image.size
        assert img.encoding == 'png'

def test_image_from_pil():
    pil_img = PILImage.new('RGB', (10, 10))
    img = Image(pil=pil_img)
    assert isinstance(img.pil, PILImage.Image)
    assert img.size == (10, 10)
    
    img = Image(pil_img)
    assert isinstance(img.pil, PILImage.Image)

def test_image_load_image_from_base64(image):
    data_url = image.url
    pil_img = Image.load_image(data_url)
    assert isinstance(pil_img, PILImage.Image)


def test_image_validate_kwargs_with_array():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    validated_values = Image.validate_kwargs({'array': arr})
    assert isinstance(validated_values['pil'], PILImage.Image)
    assert validated_values['size'] == (10, 10)

def test_image_validate_kwargs_with_invalid_encoding():
    with pytest.raises(ValueError):
        Image.validate_kwargs({'encoding': 'invalid_format'})

# Invoke pytest
if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
