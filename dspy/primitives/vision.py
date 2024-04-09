import base64 as base64lib
import io
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
import PIL.Image as PILImage
from pydantic import BaseModel, ConfigDict, Field, model_validator

SupportsImage = Union[str, 'Image', np.ndarray, PILImage.Image, Path]
SupportsPrompt = Union[str, SupportsImage]


class Image(BaseModel):
  """A class to represent an image. The image can be initialized with a numpy array, a base64 string, or a file path.

  Attributes:
      array (Optional[np.ndarray]): The image represented as a NumPy array.
      base64 (str): The image encoded as a base64 string.
      encoding (str): The format used for encoding the image when converting to base64.
      path (Optional[str]): The file path to the image if initialized from a file.
      pil (Optional[PILImage.Image]): The image represented as a PIL Image object.
      url (Optional[str]): The URL to the image if initialized from a URL.
      size (Optional[tuple[int, int]]): The size of the image as a (width, height) tuple.

  Example:
      >>> from vision import Image
      >>> import numpy as np
      >>> # Initialize with a NumPy array
      >>> arr = np.zeros((100, 100, 3), dtype=np.uint8)
      >>> img_from_array = Image(arr)
      >>> # Initialize with a base64 string
      >>> base64_str = 'iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=='
      >>> img_from_base64 = Image(base64_str)
      >>> # Initialize with a file path
      >>> img_from_path = Image('path/to/image.png')
      >>> # Access the PIL Image object
      >>> pil_image = img_from_array.pil
  """
  model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)
  
  array: Optional[np.ndarray] = Field(None, exclude=True)
  base64: Optional[str] = ''
  encoding: str = 'png'
  path: Optional[str] = ''
  pil: Optional[PILImage.Image] = Field(None, exclude=True)
  url: Optional[str] = Field(None, exclude=True)
  size: Optional[tuple[int, int]] = Field(None, exclude=True)

  def __init__(self, arg=None, **kwargs):
    if arg is not None:
      if isinstance(arg, str):
        if urlparse(arg).scheme:
          kwargs['url'] = arg
        elif Path(arg).is_file():
          kwargs['path'] = arg
        else:
          kwargs['base64'] = arg
      elif isinstance(arg, Path):
        kwargs['path'] = str(arg)
      elif isinstance(arg, np.ndarray):
        kwargs['array'] = arg
      elif isinstance(arg, PILImage.Image):
        kwargs['pil'] = arg
      else:
        raise ValueError(f"Unsupported argument type '{type(arg)}'.")
    super().__init__(**kwargs)

  def __repr__(self):
    """Return a string representation of the image."""
    return f"Image(base64={self.base64[:10]}..., encoding={self.encoding}, size={self.size})"


  @staticmethod
  def from_pil(image: PILImage.Image, encoding: str = 'png') -> 'Image':
      """Creates an Image instance from a PIL image.

      Args:
          image (PIL.Image.Image): The source PIL image from which to create the Image instance.
          encoding (str): The format used for encoding the image when converting to base64.

      Returns:
          Image: An instance of the Image class with populated fields.
      """
      buffer = io.BytesIO()
      image_format = image.format or encoding.upper()
      image.save(buffer, format=image_format)
      base64_encoded = base64lib.b64encode(buffer.getvalue()).decode('utf-8')
      data_url = f"data:image/{encoding};base64,{base64_encoded}"

      return {
          'array': np.array(image),
          'base64': base64_encoded,
          'pil': image,
          'size': image.size,
          'url': data_url,
      }

  @staticmethod
  def load_image(url: str) -> PILImage.Image:
      """Downloads an image from a URL or decodes it from a base64 data URI.

      Args:
          url (str): The URL of the image to download, or a base64 data URI.

      Returns:
          PIL.Image.Image: The downloaded and decoded image as a PIL Image object.
      """
      if url.startswith('data:image'):
          # Extract the base64 part of the data URI
          base64_str = url.split(';base64', 1)[1]
          image_data = base64lib.b64decode(base64_str)
      else:
          # Open the URL and read the image data
          with urlopen(url) as response:
              image_data = response.read()

      # Convert the image data to a PIL Image
      buffer = io.BytesIO(image_data)
      return PILImage.frombuffer('RGB', (10, 10), image_data)
    
  @model_validator(mode='before')
  @classmethod
  def validate_kwargs(cls, values) -> dict:
      """Validates and transforms input data before model initialization.

      Ensures that all values are not None and are consistent.

      Args:
          values (dict): The input data to validate.

      Returns:
          dict: The validated and possibly transformed input data.
      """
      # Check for mutually exclusive fields
      image_fields = ['array', 'base64', 'path', 'pil', 'url']
      provided_fields = [field for field in image_fields if field in values]
      if len(provided_fields) > 1:
          raise ValueError(f"Only one of {image_fields} should be provided.")

      # Initialize all fields to None or their default values
      validated_values = {
          'array': None,
          'base64': '',
          'encoding': values.get('encoding', 'jpeg'),
          'path': None,
          'pil': None,
          'url': None,
          'size': None,
      }

      # Load the image and populate other fields
      if 'path' in values:
          image = PILImage.open(values['path'])
          validated_values.update(cls.from_pil(image, validated_values['encoding']))
          validated_values['path'] = values['path']

      # Convert to PIL image and populate other fields
      elif 'array' in values:
          image = PILImage.fromarray(values['array'])
          validated_values.update(cls.from_pil(image, validated_values['encoding']))

      elif 'pil' in values:
          validated_values.update(cls.from_pil(values['pil'], validated_values['encoding']))

      # If 'base64' is provided, decode and populate other fields
      elif 'base64' in values:
          image_data = base64lib.b64decode(values['base64'])
          image = PILImage.open(io.BytesIO(image_data))
          validated_values.update(cls.from_pil(image, validated_values['encoding']))

      # If 'url' is provided, download the image and populate other fields
      elif 'url' in values:
        image = cls.load_image(values['url'])
        
        # Determine the encoding based on the URL's file extension
        url_path = urlparse(values['url']).path
        file_extension = Path(url_path).suffix[1:].lower()
        validated_values['encoding'] = file_extension
        validated_values.update(cls.from_pil(image, validated_values['encoding']))
        validated_values['url'] = values['url']
        
      if validated_values['encoding'] not in ['png', 'jpeg', 'jpg', 'bmp', 'gif']:
          raise ValueError("The 'encoding' must be a valid image format (png, jpeg, jpg, bmp, gif).")

      return validated_values

  def save(self, path: str) -> None:
    self.pil.save(path)

  def model_dump(self) -> dict:
    return {'base64': self.base64, 'encoding': self.encoding, 'size': self.size}
