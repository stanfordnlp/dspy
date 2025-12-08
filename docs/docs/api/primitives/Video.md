# dspy.Video

The `dspy.Video` type enables video understanding capabilities in DSPy, with native support for Google's Gemini models via LiteLLM.

## Overview

`dspy.Video` supports multiple input sources:

- **Local files**: Automatically encoded to base64 (for files under 20MB)
- **Remote URLs**: HTTP(S) video URLs passed directly to the model
- **YouTube URLs**: Native Gemini support for YouTube videos
- **Pre-uploaded files**: Reference videos uploaded via Gemini's Files API
- **Raw bytes**: Video data with specified MIME type

## Supported Formats

MP4, MPEG, MOV, AVI, FLV, MPG, WebM, WMV, and 3GPP.

## Basic Usage

### From a Local File

```python
import dspy

# Simple construction
video = dspy.Video("./my_video.mp4")

# Or using the factory method
video = dspy.Video.from_path("./my_video.mp4")
```

### From a URL

```python
# Remote video URL
video = dspy.Video("https://example.com/video.mp4")

# Or explicitly
video = dspy.Video.from_url("https://example.com/video.mp4")
```

### From YouTube

```python
# YouTube videos are natively supported by Gemini
video = dspy.Video.from_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# Or just pass the URL directly
video = dspy.Video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
```

### From a Pre-uploaded File ID

For videos larger than 20MB, upload them first using Gemini's Files API:

```python
# Reference an already-uploaded file
video = dspy.Video.from_file_id("files/abc123", mime_type="video/mp4")
```

## Using Video in Signatures

### Basic Video Question-Answering

```python
import dspy

class VideoQA(dspy.Signature):
    """Answer questions about a video."""
    video: dspy.Video = dspy.InputField(desc="The video to analyze")
    question: str = dspy.InputField(desc="Question about the video")
    answer: str = dspy.OutputField(desc="Answer based on the video content")

# Configure with a Gemini model
lm = dspy.LM("gemini/gemini-2.0-flash")
dspy.configure(lm=lm)

# Use the signature
qa = dspy.Predict(VideoQA)
result = qa(
    video=dspy.Video("./clip.mp4"),
    question="What is happening in this video?"
)
print(result.answer)
```

### Video Summarization

```python
class VideoSummary(dspy.Signature):
    """Generate a summary of the video content."""
    video: dspy.Video = dspy.InputField()
    summary: str = dspy.OutputField(desc="A concise summary of what happens in the video")

summarize = dspy.ChainOfThought(VideoSummary)
result = summarize(video=dspy.Video("https://example.com/presentation.mp4"))
```

### Multiple Videos

```python
class VideoComparison(dspy.Signature):
    """Compare multiple videos."""
    videos: list[dspy.Video] = dspy.InputField(desc="Videos to compare")
    comparison: str = dspy.OutputField(desc="Comparison of the videos")

compare = dspy.Predict(VideoComparison)
result = compare(videos=[
    dspy.Video("./video1.mp4"),
    dspy.Video("./video2.mp4"),
])
```

## Handling Large Videos (>20MB)

For videos exceeding the 20MB inline limit, use the Gemini Files API:

```python
import os

# Set your API key
os.environ["GEMINI_API_KEY"] = "your-api-key"

# Create video from large file (will raise error if used directly)
video = dspy.Video.from_path("./large_video.mp4")

# Upload to Gemini Files API
uploaded_video = video.upload()

# Now use the uploaded video (references file_id)
result = qa(video=uploaded_video, question="What happens at the end?")
```

You can also pass the API key directly:

```python
uploaded_video = video.upload(api_key="your-api-key")
```

## Combining with Other Media Types

```python
class MultiModalAnalysis(dspy.Signature):
    """Analyze video and image together."""
    video: dspy.Video = dspy.InputField()
    thumbnail: dspy.Image = dspy.InputField()
    analysis: str = dspy.OutputField()

analyze = dspy.Predict(MultiModalAnalysis)
result = analyze(
    video=dspy.Video("./clip.mp4"),
    thumbnail=dspy.Image("./thumbnail.jpg")
)
```

## Model Compatibility

Video understanding is currently best supported by **Google Gemini** models:

- `gemini/gemini-2.0-flash`
- `gemini/gemini-2.0-pro`
- `gemini/gemini-1.5-flash`
- `gemini/gemini-1.5-pro`

Other models may have limited or no video support.

## Technical Details

- **Inline size limit**: 20MB (larger files require Files API upload)
- **Frame sampling**: Gemini samples at 1 FPS by default
- **Token usage**: ~300 tokens per second of video at default resolution
- **Video duration**: Up to 2 hours at default resolution, 6 hours at low resolution
- **YouTube limits**: Free tier allows 8 hours/day; paid tiers unlimited

<!-- START_API_REF -->
::: dspy.Video
    handler: python
    options:
        members:
            - adapt_to_native_lm_feature
            - description
            - extract_custom_type_from_annotation
            - format
            - from_bytes
            - from_file_id
            - from_path
            - from_url
            - from_youtube
            - is_streamable
            - parse_lm_response
            - parse_stream_chunk
            - serialize_model
            - upload
        show_source: true
        show_root_heading: true
        heading_level: 2
        docstring_style: google
        show_root_full_path: true
        show_object_full_path: false
        separate_signature: false
        inherited_members: true
:::
<!-- END_API_REF -->
