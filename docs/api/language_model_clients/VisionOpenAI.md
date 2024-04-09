
---
sidebar_position: 1
---

# dspy.GPT4Vision

### Usage

```python
lm = dspy.GPT4Vision(model='gpt-4-vision-preview', api_key='your_api_key')
```

### Constructor

The constructor initializes the base class `LM` and sets up the necessary attributes for making requests to the GPT-4 Vision API. It takes the following parameters:

```python
class GPT4Vision(LM):
    def __init__(
        self,
        model: str = "gpt-4-vision-preview",
        api_key: Optional[str] = None,
        **kwargs,
    ):
```

**Parameters:**
- `model` (_str_, _optional_): The name of the GPT-4 Vision model to use. Defaults to "gpt-4-vision-preview".
- `api_key` (_Optional[str]_, _optional_): The API key for authentication. Defaults to None.
- `**kwargs`: Additional keyword arguments to pass to the API.

### Methods

#### `encode_image_to_base64(self, image_path: str) -> str`

Encodes an image file to a base64 string.

**Parameters:**
- `image_path` (_str_): The path to the image file.

**Returns:**
- `str`: The base64-encoded string of the image.

#### `prepare_image_data(self, images: List[dict], is_url: bool = True) -> List[dict]`

Prepares the image data for the API call.

**Parameters:**
- `images` (_List[dict]_): A list of dictionaries containing image data.
- `is_url` (_bool_, _optional_): Indicates whether the image data is provided as URLs or base64-encoded strings. Defaults to True.

**Returns:**
- `List[dict]`: A list of dictionaries containing the prepared image data.

#### `calculate_image_tokens(self, images: List[dict]) -> int`

Calculates the token cost of the images.

**Parameters:**
- `images` (_List[dict]_): A list of dictionaries containing image data.

**Returns:**
- `int`: The total token cost of the images.

#### `basic_request(self, prompt: Union[str, List[dict]], **kwargs) -> Any`

Makes a basic request to the GPT-4 Vision model, supporting text and image prompts.

**Parameters:**
- `prompt` (_Union[str, List[dict]]_): The prompt or list of prompts to send to the API.
- `**kwargs`: Additional keyword arguments to pass to the API.

**Returns:**
- `Any`: The API response.

#### `__call__(self, prompts: List[dict], **kwargs) -> List[Any]`

Retrieves responses from GPT-4 Vision based on the prompts.

**Parameters:**
- `prompts` (_List[dict]_): A list of prompts with text or images.
- `**kwargs`: Additional keyword arguments to pass to the API.

**Returns:**
- `List[Any]`: A list of responses from the model.

### Error Handling

The `GPT4Vision` class includes error handling for various scenarios:

- If the `openai` module is not available, an `ImportError` is raised.
- If an error occurs during the API request, a `requests.exceptions.RequestException` is raised.
- If the API returns a non-200 status code, appropriate exceptions are raised based on the status code.

### Logging

The `GPT4Vision` class uses the `logging` module to log errors and exceptions. The logs are stored in the specified log file.

### Requirements

- Python 3.x
- `requests` library
- `openai` library (optional, but required for making API requests)

Make sure to install the required dependencies before using the `GPT4Vision` class.
