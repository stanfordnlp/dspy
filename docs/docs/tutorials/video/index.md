# Video Understanding with Gemini

This tutorial demonstrates how to use `dspy.Video` to build video understanding applications with Google's Gemini models. Video understanding enables powerful use cases like video Q&A, content summarization, scene detection, and more.

## Prerequisites

- A Google AI Studio API key (get one at [aistudio.google.com](https://aistudio.google.com))
- DSPy installed with LiteLLM support

```bash
pip install dspy-ai
```

## Setup

First, configure DSPy with a Gemini model:

```python
import dspy
import os

os.environ["GEMINI_API_KEY"] = "your-api-key-here"

# Configure with Gemini 2.0 Flash (recommended for video)
lm = dspy.LM("gemini/gemini-2.0-flash")
dspy.configure(lm=lm)
```

## Basic Video Question-Answering

The simplest use case is asking questions about a video:

```python
class VideoQA(dspy.Signature):
    """Answer questions about a video."""
    video: dspy.Video = dspy.InputField(desc="The video to analyze")
    question: str = dspy.InputField(desc="Question about the video")
    answer: str = dspy.OutputField(desc="Answer based on the video content")

qa = dspy.Predict(VideoQA)

# From a local file
result = qa(
    video=dspy.Video("./my_video.mp4"),
    question="What is happening in this video?"
)
print(result.answer)
```

## Different Video Sources

`dspy.Video` supports multiple input sources:

### Local Files

```python
# Direct path
video = dspy.Video("./path/to/video.mp4")

# Using factory method
video = dspy.Video.from_path("./path/to/video.mp4")
```

### Remote URLs

```python
# HTTP/HTTPS URLs
video = dspy.Video("https://example.com/video.mp4")

# Using factory method
video = dspy.Video.from_url("https://example.com/video.mp4")
```

### YouTube Videos

Gemini natively supports YouTube URLs - no download required:

```python
# YouTube watch URLs
video = dspy.Video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# Short URLs
video = dspy.Video("https://youtu.be/dQw4w9WgXcQ")

# Using factory method
video = dspy.Video.from_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
```

### Pre-uploaded Files (Gemini Files API)

For videos you've already uploaded:

```python
video = dspy.Video.from_file_id("files/abc123", mime_type="video/mp4")
```

## Video Summarization

Create summaries of video content:

```python
class VideoSummary(dspy.Signature):
    """Generate a detailed summary of video content."""
    video: dspy.Video = dspy.InputField(desc="The video to summarize")
    summary: str = dspy.OutputField(desc="Comprehensive summary of the video")

summarize = dspy.ChainOfThought(VideoSummary)
result = summarize(video=dspy.Video("./presentation.mp4"))
print(result.summary)
```

## Timestamp-Based Analysis

Ask about specific moments in a video:

```python
class TimestampAnalysis(dspy.Signature):
    """Analyze specific moments in a video."""
    video: dspy.Video = dspy.InputField()
    question: str = dspy.InputField()
    timestamp: str = dspy.OutputField(desc="Timestamp in MM:SS format")
    description: str = dspy.OutputField(desc="What happens at this moment")

analyze = dspy.Predict(TimestampAnalysis)
result = analyze(
    video=dspy.Video("./tutorial.mp4"),
    question="When does the speaker introduce the main topic?"
)
print(f"At {result.timestamp}: {result.description}")
```

## Working with Multiple Videos

Compare or analyze multiple videos together:

```python
class VideoComparison(dspy.Signature):
    """Compare multiple videos."""
    videos: list[dspy.Video] = dspy.InputField(desc="Videos to compare")
    aspect: str = dspy.InputField(desc="What aspect to compare")
    comparison: str = dspy.OutputField(desc="Detailed comparison")

compare = dspy.Predict(VideoComparison)
result = compare(
    videos=[
        dspy.Video("./video1.mp4"),
        dspy.Video("./video2.mp4"),
    ],
    aspect="presentation style and clarity"
)
print(result.comparison)
```

## Combining Video with Other Inputs

### Video + Text Context

```python
class VideoWithContext(dspy.Signature):
    """Analyze video with additional context."""
    video: dspy.Video = dspy.InputField()
    context: str = dspy.InputField(desc="Background information")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

analyze = dspy.Predict(VideoWithContext)
result = analyze(
    video=dspy.Video("./product_demo.mp4"),
    context="This is a demo of our new software product targeting enterprise customers.",
    question="What features are highlighted and how well are they explained?"
)
```

### Video + Image (Thumbnail Analysis)

```python
class VideoWithThumbnail(dspy.Signature):
    """Analyze video and its thumbnail."""
    video: dspy.Video = dspy.InputField()
    thumbnail: dspy.Image = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

analyze = dspy.Predict(VideoWithThumbnail)
result = analyze(
    video=dspy.Video("./video.mp4"),
    thumbnail=dspy.Image("./thumbnail.jpg"),
    question="Does the thumbnail accurately represent the video content?"
)
```

## Handling Large Videos (>20MB)

For videos larger than 20MB, you need to upload them to Gemini's Files API first:

```python
# This will raise an error for large files
# video = dspy.Video("./large_video.mp4")

# Instead, upload first
video = dspy.Video.from_path("./large_video.mp4")
uploaded_video = video.upload()  # Uploads to Gemini Files API

# Now use the uploaded video
result = qa(video=uploaded_video, question="What happens in this video?")
```

You can also pass the API key explicitly:

```python
uploaded_video = video.upload(api_key="your-api-key")
```

## Building a Video Content Moderator

Here's a practical example of a content moderation system:

```python
from typing import Literal

class ContentModeration(dspy.Signature):
    """Analyze video for content policy violations."""
    video: dspy.Video = dspy.InputField(desc="Video to moderate")

    is_safe: bool = dspy.OutputField(desc="Whether the content is safe")
    category: Literal["safe", "violence", "adult", "hate_speech", "other"] = dspy.OutputField()
    confidence: Literal["high", "medium", "low"] = dspy.OutputField()
    explanation: str = dspy.OutputField(desc="Brief explanation of the decision")

moderate = dspy.ChainOfThought(ContentModeration)

def check_video(video_path: str) -> dict:
    result = moderate(video=dspy.Video(video_path))
    return {
        "is_safe": result.is_safe,
        "category": result.category,
        "confidence": result.confidence,
        "explanation": result.explanation
    }

# Usage
moderation_result = check_video("./user_upload.mp4")
if not moderation_result["is_safe"]:
    print(f"Content flagged: {moderation_result['category']}")
    print(f"Reason: {moderation_result['explanation']}")
```

## Building a Video Search System

Create a system that finds relevant moments in videos:

```python
import json

class VideoSearch(dspy.Signature):
    """Find moments in a video matching a query."""
    video: dspy.Video = dspy.InputField()
    query: str = dspy.InputField(desc="What to search for in the video")

    found: bool = dspy.OutputField(desc="Whether matching content was found")
    moments: str = dspy.OutputField(desc="JSON list of {timestamp, description} objects")

search = dspy.Predict(VideoSearch)

def find_moments(video_path: str, query: str) -> list[dict]:
    result = search(video=dspy.Video(video_path), query=query)
    if result.found:
        return json.loads(result.moments)
    return []

# Usage
moments = find_moments("./lecture.mp4", "when the professor explains machine learning")
for moment in moments:
    print(f"{moment['timestamp']}: {moment['description']}")
```

## Optimizing Video Programs with DSPy

You can optimize your video programs just like any other DSPy program:

```python
from dspy.teleprompt import BootstrapFewShot

# Define your signature
class VideoClassification(dspy.Signature):
    """Classify video content type."""
    video: dspy.Video = dspy.InputField()
    category: Literal["tutorial", "entertainment", "news", "sports", "other"] = dspy.OutputField()

# Create training examples
trainset = [
    dspy.Example(
        video=dspy.Video("./examples/tutorial1.mp4"),
        category="tutorial"
    ).with_inputs("video"),
    dspy.Example(
        video=dspy.Video("./examples/news1.mp4"),
        category="news"
    ).with_inputs("video"),
    # ... more examples
]

# Create and optimize
classifier = dspy.Predict(VideoClassification)
optimizer = BootstrapFewShot(metric=lambda example, pred, trace: example.category == pred.category)
optimized_classifier = optimizer.compile(classifier, trainset=trainset)
```

## Best Practices

1. **Choose the right video source**: Use YouTube URLs when possible (no upload needed). Use local files for private content.

2. **Keep videos concise**: Gemini processes videos at 1 FPS. Longer videos use more tokens (~300 tokens/second).

3. **Be specific in questions**: Instead of "What's in this video?", ask "What product features are demonstrated in the first 30 seconds?"

4. **Use ChainOfThought for complex analysis**: For nuanced tasks, `dspy.ChainOfThought` helps the model reason through the video content.

5. **Handle large files appropriately**: Always use `upload()` for videos over 20MB to avoid errors.

6. **Consider token limits**: A 2-minute video uses ~36,000 tokens. Plan your context budget accordingly.

## Supported Models

Video understanding works best with these Gemini models:

| Model | Video Support | Notes |
|-------|--------------|-------|
| `gemini/gemini-2.0-flash` | Full | Recommended for most use cases |
| `gemini/gemini-2.0-pro` | Full | Better quality, higher latency |
| `gemini/gemini-1.5-flash` | Full | Good balance of speed/quality |
| `gemini/gemini-1.5-pro` | Full | Highest quality |

## Limitations

- **Inline size limit**: 20MB (use Files API for larger videos)
- **Maximum duration**: 2 hours at default resolution
- **Frame sampling**: 1 FPS (fast action may lose detail)
- **YouTube free tier**: 8 hours of video per day
- **Provider support**: Currently Gemini-only

## Next Steps

- Check out the [API Reference](/api/primitives/Video) for complete method documentation
- Learn about [Signatures](/learn/programming/signatures) for more complex use cases
- Explore [Optimizers](/learn/optimization/optimizers) to improve your video programs
